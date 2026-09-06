#!/usr/bin/env python3
"""
moe_ref.py: an expert-parallel MoE layer in PyTorch, the reference the Charm++
example in ../moe.C is compared against.

Same layer, routing, arithmetic and checksums as the example (the routing hash,
the init hash, the chunked SGD order and the loss are ports of moe.C/moe.cu),
written the way an MoE training framework writes it:

  - one process per GPU, torch.distributed over NCCL;
  - dispatch and combine are one all_to_all_single each, every token bundled
    per destination rank (not one message per expert);
  - the host runs a step ahead of the device (double-buffered slabs, the next
    step's token counts exchanged before this step's combine is issued);
  - load balancing by expert re-placement between steps, in two flavours:
      --lb sync   EPLB-style: at an LB step the placement is recomputed from
                  the step's token counts and the weights are moved with NCCL
                  point-to-point sends before the step starts (a full stop);
      --lb async  the same decision, but the weights move after the old
                  owner's compute of that step, overlapped with the rest of
                  the step, and the placement takes effect at the next step.
    --place greedy moves as few experts as needed (pairwise transfers from
    the most to the least loaded rank); --place lpt repacks all experts
    largest-first with an equal count per rank, EPLB's packing.

Every Step line matches the example's format so the same scripts parse both.
Checksums: out2 is the sum of squares of every rank's combined output, w2 the
sum of squares of every expert's W1 and W2 after the step's update, both in
double; they agree with the example's to fp32 accumulation-order noise.
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# ---------------------------------------------------------------- hashing
# Ports of sm64/hmix/u01 (moe.C, host) and splitmix64/u01 (moe.cu, device).
MASK = (1 << 64) - 1
C1 = 0x9E3779B97F4A7C15
C2 = 0xBF58476D1CE4E5B9
C3 = 0x94D049BB133111EB
C4 = 0x632BE59BD9B4E019
U = np.uint64


def sm64(x):
    x = (x + C1) & MASK
    x = ((x ^ (x >> 30)) * C2) & MASK
    x = ((x ^ (x >> 27)) * C3) & MASK
    return x ^ (x >> 31)


def hmix(a, b):
    return sm64(a ^ ((b * C1 + C4) & MASK))


def sm64_np(x):
    x = x + U(C1)
    x = (x ^ (x >> U(30))) * U(C2)
    x = (x ^ (x >> U(27))) * U(C3)
    return x ^ (x >> U(31))


def hmix_np(a, b):
    return sm64_np(a ^ (b * U(C1) + U(C4)))


def u01_np(h):
    return ((h >> U(11)).astype(np.float64) + 0.5) / 9007199254740992.0


def i64(u):
    """The int64 with the same bits as the uint64 u (torch has no uint64 math)."""
    return u - (1 << 64) if u >= (1 << 63) else u


def lsr(x, s):
    """Logical right shift of an int64 tensor holding uint64 bits."""
    return (x >> s) & ((1 << (64 - s)) - 1)


def splitmix64_t(x):
    x = x + i64(C1)
    x = (x ^ lsr(x, 30)) * i64(C2)
    x = (x ^ lsr(x, 27)) * i64(C3)
    return x ^ lsr(x, 31)


def init_uniform(n, seed, bound, device, dtype=torch.float32, block=1 << 24):
    """moeInitUniform: p[i] = bound * (2 u - 1), u from two splitmix rounds of
    (seed, i). Computed in blocks so the int64 temporaries stay small."""
    out = torch.empty(n, dtype=dtype, device=device)
    for i0 in range(0, n, block):
        i = torch.arange(i0, min(n, i0 + block), dtype=torch.int64, device=device)
        h = splitmix64_t((i * i64(C1)) ^ i64(seed))
        h = splitmix64_t(h + i)
        v = lsr(h, 40).to(torch.float32)
        u = (v + 0.5) * (1.0 / 16777216.0)
        out[i0:i0 + i.numel()] = (bound * (2.0 * u - 1.0)).to(dtype)
    return out


# ---------------------------------------------------------------- routing
class Router:
    """Dispatcher::route(): a pure function of (seed, step, pe, token, slot)."""

    def __init__(self, E, T, K, zipf, drift, seed, pe):
        self.E, self.T, self.K = E, T, K
        self.zipf, self.drift, self.seed, self.pe = zipf, drift, seed, pe
        self.cur_phase = -1

    def build_phase(self, phase):
        E = self.E
        perm = list(range(E))
        for i in range(E - 1, 0, -1):
            h = hmix(hmix(self.seed, 77), phase * E + i)
            j = h % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        cdf = []
        s = 0.0
        for r in range(E):
            s += math.pow(r + 1, -self.zipf) if self.zipf > 0.0 else 1.0
            cdf.append(s)
        self.cdf = np.array([c / s for c in cdf], dtype=np.float64)
        self.perm = np.array(perm, dtype=np.int64)

    def route(self, step):
        E, T, K = self.E, self.T, self.K
        phase = (step - 1) // self.drift if self.drift > 0 else 0
        if phase != self.cur_phase:
            self.cur_phase = phase
            self.build_phase(phase)
        base = hmix(hmix(self.seed, step), self.pe)
        with np.errstate(over="ignore"):
            ht = hmix_np(U(base), np.arange(T, dtype=U))
            chosen = np.full((T, K), -1, dtype=np.int64)
            for j in range(K):
                hj = hmix_np(ht, U(j))
                pending = np.ones(T, dtype=bool)
                for attempt in range(16):
                    idx = np.flatnonzero(pending)
                    if idx.size == 0:
                        break
                    h = hmix_np(hj[idx], U(attempt))
                    r = np.searchsorted(self.cdf, u01_np(h), side="right")
                    cand = self.perm[np.minimum(r, E - 1)]
                    ok = np.ones(idx.size, dtype=bool)
                    for q in range(j):
                        ok &= chosen[idx, q] != cand
                    chosen[idx[ok], j] = cand[ok]
                    pending[idx[ok]] = False
                for ti in np.flatnonzero(pending):
                    e = (chosen[ti, j - 1] + 1) % E
                    while e in chosen[ti, :j]:
                        e = (e + 1) % E
                    chosen[ti, j] = e
        slot_expert = chosen.reshape(-1)  # (token, slot) order
        counts = np.bincount(slot_expert, minlength=E).astype(np.int64)
        return slot_expert, counts


# ---------------------------------------------------------------- placement
def place_greedy(owner, load, P, tol=0.02):
    """Pairwise transfers from the most to the least loaded rank until the gap
    is within tol of the average; each move takes the expert whose load is
    closest to half the gap. Few moves, like a diffusion balancer."""
    owner = owner.copy()
    R = np.bincount(owner, weights=load, minlength=P)
    avg = load.sum() / P
    while True:
        hi, lo = int(R.argmax()), int(R.argmin())
        gap = R[hi] - R[lo]
        if gap <= tol * avg:
            break
        cands = np.flatnonzero(owner == hi)
        cands = cands[load[cands] < gap]
        if cands.size == 0:
            break
        e = int(cands[np.argmin(np.abs(load[cands] - gap / 2))])
        owner[e] = lo
        R[hi] -= load[e]
        R[lo] += load[e]
    return owner


def place_lpt(owner, load, P):
    """EPLB's packing: experts largest-first onto the least loaded rank with
    room, an equal number per rank; rank labels then permuted to keep as many
    experts in place as possible."""
    E = len(load)
    cap = (E + P - 1) // P
    order = np.argsort(-load, kind="stable")
    R = np.zeros(P)
    cnt = np.zeros(P, dtype=np.int64)
    new = np.zeros(E, dtype=np.int64)
    for e in order:
        r = int(np.argmin(np.where(cnt < cap, R, np.inf)))
        new[e] = r
        R[r] += load[e]
        cnt[r] += 1
    best, best_keep = None, -1
    for perm in itertools.permutations(range(P)):
        relab = np.array(perm)[new]
        keep = int((relab == owner).sum())
        if keep > best_keep:
            best, best_keep = relab, keep
    return best


# ---------------------------------------------------------------- the layer
class Lane:
    def __init__(self, chunk, dff, dtype, device, adam, dm, fuse_rows):
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.h = torch.empty(chunk, dff, dtype=dtype, device=device)
            self.dh = torch.empty(chunk, dff, dtype=dtype, device=device)
            self.grads = None
            if adam:
                self.grads = (torch.zeros(dm, dff, dtype=dtype, device=device),
                              torch.zeros(dff, dm, dtype=dtype, device=device))
            self.fx = self.fy = None
            self.fuse = fuse_rows
            self.dm, self.dtype, self.device = dm, dtype, device

    def fused(self, rows):
        """Scratch for one expert's tokens, grown on demand: the worst case
        (every token of every rank on one expert) is several GB at Llama size
        and never happens."""
        if self.fx is None or self.fx.shape[0] < rows:
            n = max(rows, 2 * (0 if self.fx is None else self.fx.shape[0]))
            self.fx = torch.empty(n, self.dm, dtype=self.dtype, device=self.device)
            self.fy = torch.empty(n, self.dm, dtype=self.dtype, device=self.device)
        return self.fx[:rows], self.fy[:rows]


def expert_chunks(lane, W1, W2, x, y, chunk, lr, scale, grads=None):
    """moeExpertChunks: per chunk h = relu(x W1), y = h W2, dh = y W2^T,
    dz = dh * (h > 0), W2 -= lr scale h^T y, W1 -= lr scale x^T dz (the dh
    GEMM reads the pre-update W2); with grads=(dW1, dW2) the gradients are
    accumulated instead (Adam)."""
    n = x.shape[0]
    for c0 in range(0, n, chunk):
        nc = min(chunk, n - c0)
        xc = x[c0:c0 + nc]
        yc = y[c0:c0 + nc]
        h = lane.h[:nc]
        dh = lane.dh[:nc]
        torch.mm(xc, W1, out=h)
        h.relu_()
        torch.mm(h, W2, out=yc)
        torch.mm(yc, W2.t(), out=dh)
        dh.masked_fill_(h <= 0, 0.0)
        if grads is None:
            W2.addmm_(h.t(), yc, alpha=-lr * scale)
            W1.addmm_(xc.t(), dh, alpha=-lr * scale)
        else:
            grads[1].addmm_(h.t(), yc, alpha=scale)
            grads[0].addmm_(xc.t(), dh, alpha=scale)


def adam_apply(W, m, v, g, lr, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    c1 = 1.0 / (1.0 - b1 ** t)
    c2 = 1.0 / (1.0 - b2 ** t)
    m.mul_(b1).add_(g, alpha=1.0 - b1)
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    W.addcdiv_(m * c1, (v * c2).sqrt_().add_(eps), value=-lr)


class Expert:
    __slots__ = ("W1", "W2", "m1", "v1", "m2", "v2", "adam_t")

    def __init__(self):
        self.W1 = self.W2 = None
        self.m1 = self.v1 = self.m2 = self.v2 = None
        self.adam_t = 0

    def tensors(self, adam):
        ts = [self.W1, self.W2]
        if adam:
            ts += [self.m1, self.v1, self.m2, self.v2]
        return ts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experts", type=int, default=64)
    ap.add_argument("-m", "--d-model", type=int, default=2048)
    ap.add_argument("-H", "--d-ff", type=int, default=8192)
    ap.add_argument("-t", "--tokens", type=int, default=8192, help="per rank per step")
    ap.add_argument("-k", "--topk", type=int, default=1)
    ap.add_argument("-i", "--steps", type=int, default=50, help="timed steps")
    ap.add_argument("-u", "--warmup", type=int, default=5)
    ap.add_argument("-z", "--zipf", type=float, default=1.0)
    ap.add_argument("-p", "--drift", type=int, default=10)
    ap.add_argument("-c", "--chunk", type=int, default=1024)
    ap.add_argument("-n", "--lanes", type=int, default=4, help="compute streams per rank")
    ap.add_argument("-L", "--lr", type=float, default=1e-4)
    ap.add_argument("-C", "--checksum", type=int, default=5, help="checksum every C steps, 0 off")
    ap.add_argument("-f", "--first-lb", type=int, default=10)
    ap.add_argument("-b", "--lb-freq", type=int, default=9999)
    ap.add_argument("-S", "--seed", type=int, default=12345)
    ap.add_argument("--lb", choices=["none", "sync", "async"], default="none")
    ap.add_argument("--place", choices=["greedy", "lpt"], default="greedy")
    ap.add_argument("-A", "--adam", action="store_true")
    ap.add_argument("-T", "--tf32", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 runs weights and activations in bf16 (throughput only)")
    ap.add_argument("--fuse", action="store_true",
                    help="one chunked pass per expert over all sources (changes chunk boundaries)")
    ap.add_argument("-P", "--print-place", action="store_true")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="synchronize the host at every step end (no run-ahead)")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", rank)))
    if torch.cuda.device_count() == 1:
        local = 0
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    P = world
    E, dm, dff, T, K = a.experts, a.d_model, a.d_ff, a.tokens, a.topk
    TK = T * K
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    esz = 2 if dtype == torch.bfloat16 else 4
    if a.adam and dtype != torch.float32:
        sys.exit("Adam needs fp32")
    if a.lb != "none" and a.first_lb <= a.warmup:
        sys.exit("the first LB step must come after the warmup")

    def log(*s):
        if rank == 0:
            print(*s, flush=True)

    # ---- placement and weights: expert e starts on rank e * P // E (block map)
    owner = np.array([(e * P) // E for e in range(E)], dtype=np.int64)
    experts = {}
    for e in range(E):
        if owner[e] == rank:
            x = Expert()
            x.W1 = init_uniform(dm * dff, hmix(hmix(a.seed, 1), e), 1.0 / math.sqrt(dm),
                                device, dtype).view(dm, dff)
            x.W2 = init_uniform(dm * dff, hmix(hmix(a.seed, 2), e), 1.0 / math.sqrt(dff),
                                device, dtype).view(dff, dm)
            if a.adam:
                x.m1, x.v1 = torch.zeros_like(x.W1), torch.zeros_like(x.W1)
                x.m2, x.v2 = torch.zeros_like(x.W2), torch.zeros_like(x.W2)
            experts[e] = x
    x_in = init_uniform(T * dm, hmix(hmix(a.seed, 1000), rank), math.sqrt(3.0),
                        device, dtype).view(T, dm)

    main = torch.cuda.current_stream()
    # A rank can receive every token of every rank in the worst case.
    cap_rows = P * TK
    lanes = [Lane(a.chunk, dff, dtype, device, a.adam, dm, a.fuse) for _ in range(a.lanes)]
    mig_stream = torch.cuda.Stream()
    # Slabs, double-buffered so the host can issue step s+1 while s runs.
    send_buf = [torch.empty(TK, dm, dtype=dtype, device=device) for _ in range(2)]
    recv_buf = [torch.empty(cap_rows, dm, dtype=dtype, device=device) for _ in range(2)]
    out_buf = [torch.empty(cap_rows, dm, dtype=dtype, device=device) for _ in range(2)]
    back_buf = [torch.empty(TK, dm, dtype=dtype, device=device) for _ in range(2)]
    y = torch.empty(T, dm, dtype=dtype, device=device)
    router = Router(E, T, K, a.zipf, a.drift, a.seed, rank)

    n_total = a.warmup + a.steps
    wbytes = esz * dm * dff * 2 * (3 if a.adam else 1)
    H = sum(math.pow(r + 1, -a.zipf) for r in range(E)) if a.zipf > 0 else E
    p_max = 1.0 / H
    log("\n[PyTorch expert-parallel MoE reference]")
    log("Experts: %d, d_model %d, d_ff %d, %s, %s%s" % (
        E, dm, dff, "Adam" if a.adam else "SGD", a.dtype, " TF32" if a.tf32 else ""))
    log("Tokens: %d per rank per step on %d ranks, top-%d, chunk %d, %d compute lanes%s%s" % (
        T, P, K, a.chunk, a.lanes, ", fused sources" if a.fuse else "",
        ", no run-ahead" if a.no_pipeline else ""))
    log("Routing: zipf %.2f (hottest expert %.1fx average), hot set reshuffled every %d steps, seed %d" % (
        a.zipf, p_max * E, a.drift, a.seed))
    log("Per expert: %.1f MB weights; torch %s, NCCL %s, NCCL_P2P_LEVEL %s" % (
        wbytes / 1e6, torch.__version__, ".".join(map(str, torch.cuda.nccl.version())),
        os.environ.get("NCCL_P2P_LEVEL", "default")))
    log("Steps: %d (+%d warmup), LB: %s%s, first LB: %d, LB frequency: %d" % (
        a.steps, a.warmup, a.lb, "" if a.lb == "none" else " (" + a.place + ")",
        a.first_lb, a.lb_freq))

    def is_lb_step(s):
        return a.lb != "none" and (s == a.first_lb or (a.lb_freq > 0 and s > a.first_lb
                                                       and s % a.lb_freq == 0))

    def migrate(moves, stream):
        """Issue NCCL point-to-point moves of expert weights on `stream`.
        Returns (requests, [(e, new tensors)] for the experts arriving here)."""
        ops = []
        arrivals = []
        for (e, src, dst) in moves:
            if src == rank:
                for t in experts[e].tensors(a.adam):
                    ops.append(dist.P2POp(dist.isend, t, dst))
            elif dst == rank:
                x = Expert()
                x.W1 = torch.empty(dm, dff, dtype=dtype, device=device)
                x.W2 = torch.empty(dff, dm, dtype=dtype, device=device)
                if a.adam:
                    x.m1, x.v1 = torch.empty_like(x.W1), torch.empty_like(x.W1)
                    x.m2, x.v2 = torch.empty_like(x.W2), torch.empty_like(x.W2)
                for t in x.tensors(a.adam):
                    ops.append(dist.P2POp(dist.irecv, t, src))
                arrivals.append((e, x))
        reqs = []
        if ops:
            with torch.cuda.stream(stream):
                reqs = dist.batch_isend_irecv(ops)
        return reqs, arrivals

    def exchange_counts(s):
        """Route step s and start the all_gather of every rank's counts."""
        t0 = time.perf_counter()
        slot_expert, counts = router.route(s)
        cnt_t = torch.from_numpy(counts).to(device)
        all_cnt = torch.empty(P, E, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(all_cnt, cnt_t)
        return slot_expert, counts, all_cnt, (time.perf_counter() - t0) * 1e3

    def layout(slot_expert, counts, owner):
        """Slab layout for one step: positions sorted by (owner, expert,
        token, slot); the per-rank split sizes."""
        key = owner[slot_expert] * E + slot_expert
        order = np.argsort(key, kind="stable")
        send_idx = order // K
        recv_pos = np.empty(TK, dtype=np.int64)
        recv_pos[order] = np.arange(TK)
        in_splits = np.bincount(owner, weights=counts, minlength=P).astype(np.int64)
        return send_idx, recv_pos, in_splits

    # ---- per-step records (event-timed on the main stream)
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(6)] for _ in range(n_total + 1)]
    step_tok = np.zeros((n_total + 1, 3))   # tokens: max per rank, max per expert, this rank
    n_moved = np.zeros(n_total + 1, dtype=np.int64)
    host_ms = np.zeros((n_total + 1, 2))    # routing, wait for the counts
    chk_out = {}
    chk_w = {}
    pending_reqs = []       # async migration in flight (waited at the next step)
    pending_owner = None    # placement to apply at the next step
    pending_moves = []
    total_moved = 0
    a2a_bytes = 0.0

    # ---- the pipeline: the counts of step s+1 are exchanged before the
    # combine of step s is issued, so the host prepares s+1 while s runs.
    nxt = exchange_counts(1)
    for s in range(1, n_total + 1):
        b = s % 2
        slot_expert, counts, all_cnt, host_ms[s, 0] = nxt

        # migrations decided last step land now
        if pending_reqs:
            for r in pending_reqs:
                r.wait()
            pending_reqs = []
        if pending_owner is not None:
            for (e, src, dst, arr) in pending_moves:
                if src == rank:
                    del experts[e]
                elif dst == rank:
                    experts[e] = arr
            owner = pending_owner
            pending_owner, pending_moves = None, []

        t0 = time.perf_counter()
        C = all_cnt.cpu().numpy()      # the host waits for the all_gather here
        host_ms[s, 1] = (time.perf_counter() - t0) * 1e3
        load = C.sum(axis=0).astype(np.float64)

        moves = []
        if is_lb_step(s):
            new_owner = (place_greedy if a.place == "greedy" else place_lpt)(owner, load, P)
            moves = [(e, int(owner[e]), int(new_owner[e])) for e in range(E)
                     if owner[e] != new_owner[e]]
            n_moved[s] = len(moves)
            total_moved += len(moves)

        ev[s][0].record(main)
        if moves and a.lb == "sync":
            reqs, arrivals = migrate(moves, main)
            for r in reqs:
                r.wait()
            for (e, src, dst) in moves:
                if src == rank:
                    del experts[e]
            for (e, x) in arrivals:
                experts[e] = x
            owner = new_owner
        ev[s][1].record(main)

        send_idx, recv_pos, in_splits = layout(slot_expert, counts, owner)
        mine = [e for e in range(E) if owner[e] == rank]
        out_splits = np.array([int(C[src][mine].sum()) for src in range(P)], dtype=np.int64)
        tot = int(out_splits.sum())
        my_tok = np.bincount(owner, weights=load, minlength=P)
        step_tok[s] = (my_tok.max(), load.max(), tot)
        if s > a.warmup:
            a2a_bytes += (tot + TK) * dm * esz

        # dispatch: gather rows in slab order, one all_to_all
        idx_t = torch.from_numpy(send_idx).to(device)
        torch.index_select(x_in, 0, idx_t, out=send_buf[b])
        work = dist.all_to_all_single(recv_buf[b][:tot], send_buf[b],
                                      output_split_sizes=out_splits.tolist(),
                                      input_split_sizes=in_splits.tolist(), async_op=True)
        work.wait()
        ev[s][2].record(main)

        # compute: segment offsets of (source, expert) in the receive slab
        seg = {}
        off = 0
        for src in range(P):
            for e in mine:
                seg[(src, e)] = off
                off += int(C[src][e])
        moving = set(e for (e, src, dst) in moves) if a.lb == "async" else set()
        order_e = sorted(mine, key=lambda e: (0 if e in moving else 1, e))
        for lane in lanes:
            lane.stream.wait_stream(main)
        mig_events = []
        for li, e in enumerate(order_e):
            lane = lanes[li % a.lanes]
            x = experts[e]
            n_tot = int(load[e])
            scale = 1.0 / n_tot if n_tot > 0 else 0.0
            g = lane.grads
            with torch.cuda.stream(lane.stream):
                if n_tot > 0:
                    if a.adam:
                        g[0].zero_()
                        g[1].zero_()
                    if a.fuse:
                        segs = [(seg[(src, e)], int(C[src][e])) for src in range(P) if C[src][e] > 0]
                        xs, ys = lane.fused(n_tot)
                        o = 0
                        for (so, n) in segs:
                            xs[o:o + n].copy_(recv_buf[b][so:so + n])
                            o += n
                        expert_chunks(lane, x.W1, x.W2, xs, ys, a.chunk, a.lr, scale, g)
                        o = 0
                        for (so, n) in segs:
                            out_buf[b][so:so + n].copy_(ys[o:o + n])
                            o += n
                    else:
                        for src in range(P):
                            n = int(C[src][e])
                            if n == 0:
                                continue
                            so = seg[(src, e)]
                            expert_chunks(lane, x.W1, x.W2, recv_buf[b][so:so + n],
                                          out_buf[b][so:so + n], a.chunk, a.lr, scale, g)
                    if a.adam:
                        x.adam_t += 1
                        adam_apply(x.W1, x.m1, x.v1, g[0], a.lr, x.adam_t)
                        adam_apply(x.W2, x.m2, x.v2, g[1], a.lr, x.adam_t)
                if e in moving:
                    evm = torch.cuda.Event()
                    evm.record(lane.stream)
                    mig_events.append(evm)
        # next step's routing and counts, queued ahead of this step's combine
        if s < n_total:
            nxt = exchange_counts(s + 1)
        if moves and a.lb == "async":
            # weights leave after their last update, overlapped with the rest
            for evm in mig_events:
                mig_stream.wait_event(evm)
            pending_reqs, arrivals = migrate(moves, mig_stream)
            arr_by_e = dict(arrivals)
            pending_moves = [(e, src, dst, arr_by_e.get(e)) for (e, src, dst) in moves]
            pending_owner = new_owner
        for lane in lanes:
            main.wait_stream(lane.stream)
        ev[s][3].record(main)

        # combine: the outputs go back the way the tokens came
        work = dist.all_to_all_single(back_buf[b], out_buf[b][:tot],
                                      output_split_sizes=in_splits.tolist(),
                                      input_split_sizes=out_splits.tolist(), async_op=True)
        work.wait()
        ev[s][4].record(main)
        pos_t = torch.from_numpy(recv_pos).to(device)
        if K == 1:
            torch.index_select(back_buf[b], 0, pos_t, out=y)
        else:
            torch.sum(back_buf[b].index_select(0, pos_t).view(T, K, dm), dim=1, out=y)
            y.mul_(1.0 / K)
        if a.checksum > 0 and s % a.checksum == 0:
            chk_out[s] = y.double().square().sum()
            wsum = torch.zeros((), dtype=torch.float64, device=device)
            for e in mine:
                wsum += experts[e].W1.double().square().sum() + experts[e].W2.double().square().sum()
            chk_w[s] = wsum
        ev[s][5].record(main)
        if s == a.warmup or a.no_pipeline:
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    ev_end = ev[n_total][5]
    if pending_reqs:
        for r in pending_reqs:
            r.wait()
        torch.cuda.synchronize()

    # ---- results: event timings from every rank, checksums summed
    ph = np.zeros((n_total + 1, 6))
    for s in range(1, n_total + 1):
        ph[s, 0] = ev[s][0].elapsed_time(ev[s][1])   # sync migration
        ph[s, 1] = ev[s][1].elapsed_time(ev[s][2])   # dispatch all_to_all
        ph[s, 2] = ev[s][2].elapsed_time(ev[s][3])   # expert compute
        ph[s, 3] = ev[s][3].elapsed_time(ev[s][4])   # combine all_to_all
        ph[s, 4] = ev[s][4].elapsed_time(ev[s][5])   # combine kernel, checksum
        ph[s, 5] = ev[s][0].elapsed_time(ev[s + 1][0]) if s < n_total else \
            ev[s][0].elapsed_time(ev_end)             # step, as the main stream saw it
    t_start = ev[a.warmup][5] if a.warmup > 0 else ev[1][0]
    total_s = t_start.elapsed_time(ev_end) / 1e3
    ph_t = torch.from_numpy(ph).to(device)
    all_ph = [torch.empty_like(ph_t) for _ in range(P)]
    dist.all_gather(all_ph, ph_t)
    all_ph = torch.stack(all_ph).cpu().numpy()      # [rank, step, phase]
    chk = {}
    if chk_out:
        ks = sorted(chk_out)
        cs = torch.stack([chk_out[k] for k in ks] + [chk_w[k] for k in ks])
        dist.all_reduce(cs)
        cs = cs.cpu().numpy()
        chk = {k: (cs[i], cs[len(ks) + i]) for i, k in enumerate(ks)}
    agg = torch.tensor([total_s, a2a_bytes], dtype=torch.float64, device=device)
    dist.all_reduce(agg, op=dist.ReduceOp.MAX)
    total_s, a2a_bytes = agg.cpu().tolist()

    if rank == 0:
        avg_tok = TK
        avg_exp = TK * P / E
        for s in range(a.warmup + 1, n_total + 1):
            comp = all_ph[:, s, 2]
            line = "Step %d: %.3f ms/step, tokens/PE max/avg %.2f, gpu/PE max %.1f ms max/avg %.2f, expert max/avg %.2f" % (
                s, all_ph[:, s, 5].max(), step_tok[s, 0] / avg_tok, comp.max(),
                comp.max() / comp.mean(), step_tok[s, 1] / avg_exp)
            if n_moved[s]:
                line += ", moved %d" % n_moved[s]
            if s in chk:
                line += ", out2 %.15e, w2 %.15e" % chk[s]
            print(line)
        timed = slice(a.warmup + 1, n_total + 1)
        names = ["migration", "dispatch a2a", "compute", "combine a2a", "combine+chk", "step"]
        means = [all_ph[:, timed, i].mean(axis=1).max() for i in range(5)]
        print("\nPhases, mean over the timed steps (ms). An all_to_all also waits for the")
        print("slowest peer, so a rank's idle time shows up in its exchange phases.")
        print("  %-14s %s" % ("rank", " ".join("%7d" % r for r in range(P))))
        for i, n in enumerate(names):
            print("  %-14s %s" % (n, " ".join("%7.1f" % v for v in all_ph[:, timed, i].mean(axis=1))))
        print("Dispatch: %.1f MB in+out per rank per step (busiest rank), %.1f GB/s" % (
            a2a_bytes / a.steps / 1e6, a2a_bytes / a.steps / 1e9 / (means[1] / 1e3)))
        print("Host per step: routing %.1f ms, wait for counts %.1f ms; experts moved: %d" % (
            host_ms[timed, 0].mean(), host_ms[timed, 1].mean(), total_moved))
        if a.print_place:
            print("Placement: " + " ".join("%d:%d" % (e, owner[e]) for e in range(E)))
        print("\nTotal time: %.3f s\nAverage step time: %.3f ms" % (total_s, total_s / a.steps * 1e3))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
