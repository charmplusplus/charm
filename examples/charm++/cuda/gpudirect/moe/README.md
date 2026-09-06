# moe — mixture-of-experts layer miniApp for GPU load balancing

A single MoE feed-forward layer, written so that the experts are migratable
Charm++ objects with their state on the GPU, and the load moves the way it does
in a real MoE: by routing. Every step each PE routes its batch of tokens across
all experts, the experts run their feed-forward on what they were sent, the
outputs go back to the tokens' owners, and the experts update their weights.
The popularity of the experts is skewed (Zipf) and the hot set is reshuffled
periodically, so which GPU is overloaded changes during the run and no static
placement can fix it.

Unlike the halo-exchange miniApps (jacobi2d, leanmd, pic2d, barnes) this
exercises

- all-to-all traffic every step (every dispatcher to every expert and back),
- objects with no position key, so DiffusionLB takes its heap path rather
  than the interval cut,
- migration payloads of 100 MB to 1.6 GB per object,
- application-declared loads instead of CUPTI measurement: an expert knows its
  cost before it runs (token count times its measured seconds per token), so
  the balancer needs no kernel tracing at all.

## Model

Expert e holds `W1 [d_model x d_ff]` and `W2 [d_ff x d_model]` in fp32 (plus
Adam moments with `-A`, tripling the payload). For a slab of n tokens it runs

    h  = relu(x W1)        y  = h W2                      forward
    dy = y                                                loss = 1/2 |y|^2
    dh = dy W2^T           dz = dh * relu'(z)             backward
    W2 -= lr h^T dy        W1 -= lr x^T dz                update

five GEMMs per chunk of `-c` tokens (cuBLAS, fp32; `-T` for TF32), so the
cost of an expert is linear in the tokens it receives and a 3:1 backward to
forward ratio like real training. The loss is self-similar so no labels are
needed; it only exists to make the weights change every step in a way that
depends on exactly which tokens each expert saw and in which order, which is
what the checksums verify.

## Decomposition and communication

- `Dispatcher` group, one per PE, never migrates. Holds `-t` tokens of
  `d_model` floats on the device (fixed for the run), routes them each step on
  the host with a hashed Zipf draw, sorts them by expert with a gather kernel,
  and ships one slab per expert with a zerocopy device send (`CkDeviceBuffer`
  and a post entry method on the expert). Expert outputs land in a receive
  slab by expert offset and a combine kernel scatters them back to token
  order (weighted `1/k` for top-k). It also owns the PE's GPU context: the one
  stream every kernel and transfer on the PE goes through, the cuBLAS handle
  with a fixed workspace, and the activation scratch.
- `Expert` 1-D chare array, `usesAtSync`. Receives one slab per dispatcher per
  step into a per-source segment (arrival order does not matter: the post
  entry method knows the source), runs the chunked step segment by segment in
  source order, sends each dispatcher its outputs from the matching output
  segment, and waits for those sends to complete before the next step.

Routing is a pure function of (seed, step, PE, token, slot). Slab order is
(token, slot) order within a source, sources are processed in PE order, chunk
boundaries fall at fixed offsets, GEMMs use a fixed workspace with atomics
disallowed, and the checksums are fixed-order double reductions. So every run
with the same arguments produces the same `out2` (sum of squares of the
combined outputs) and `w2` (sum of squares of all weights) whatever the
placement, and a migration that loses or reorders anything shows up as a
different checksum.

## Load balancing

The join point is inside the expert's step, after it has consumed every slab
of the step and before it computes (see `runStep` in `moe.ci`). Nothing can
be sent to an expert between those two events (the dispatchers are waiting
for its outputs), so a parked or migrating expert never has a device receive
landed or in flight; the runtime holds a migration until every such receive is
consumed, and an expert parked in `AtSync` with an unconsumed slab would
deadlock. `-a` uses the split barrier (`AtSyncStart` at an LB step,
`AtSyncWait` `-l` steps later) and needs `+LBAsync`.

The migratable state is the weights (and moments) plus whatever the step has
live at the moment of the move: the slabs received so far before compute, the
outputs after it. Each expert declares its load with `setObjGPUTime` every
step: its token count times an EMA of measured seconds per token (device
events around its own kernels), which is also rate-aware for free. With `-I`
the app instead switches CUPTI instrumentation on for three steps before each
LB step, as pic2d does, and the balancer uses measured loads.

## Options

    -e experts (64)        -m d_model (2048)      -h d_ff (8192)
    -t tokens per PE per step (8192)              -k top-k (1)
    -i steps (50)          -u warmup steps (5)
    -z zipf exponent (1.0; 0 = uniform)           -p hot-set reshuffle period (10; 0 = static)
    -c chunk tokens (1024) -r slab headroom (0 = auto from the zipf peak)
    -L learning rate (1e-4) -A Adam               -T TF32
    -C checksum frequency (5; 0 off)              -s stats frequency (1)
    -f first LB step (10)  -b LB frequency (9999) -a async LB  -l wait lag (3)
    -I CUPTI-measured loads instead of estimates  -S seed  -P print placement
    -U one chunked pass per source instead of one per expert (slower; the
       fused default is described under Measured below)

## Build and run

    make                                   # needs cuBLAS: MATHLIBS=<nvhpc>/math_libs/<ver>

On Delta (one A40 node, four PEs, one per GPU, NUMA-pinned by the wrapper):

    export RANKLOG_DIR=ranklogs/x; mkdir -p $RANKLOG_DIR
    srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=8 \
         --cpu-bind=none --exact --unbuffered ranklogs/moewrap.sh ./moe \
         -i 30 -u 3 -f 5 -b 5 -a +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBAsync \
         +gpushm +gpupool +gpupoolsize 1024 +gpuipceventpool 256

`ranklogs/lbcmp.sh <jobid> <reps> noLB_pool sync_pool async_pool` runs the
comparison and checks every run's checksums against the first.

Per-step output: `tokens/PE max/avg` is the routed-token imbalance across
PEs, `gpu/PE max/avg` the measured device-time imbalance (lagged one step:
an expert's time is sampled when its next step starts), `expert max/avg` the
skew across experts. With the defaults the hottest expert draws about 13x the
average.

## Measured (2026-09-06, one A40 node, four PEs, `+gpupool`)

Correctness: `out2` and `w2` are bit-identical across no-LB, sync and async
at every checkpoint, at the small size and at the defaults, with SGD and
Adam, top-1 and top-2, CUPTI-measured and estimated loads, pool and non-pool
transport, and 1 or 4 compute lanes. Migrations happened in every LB run.

Step time at the defaults (64 experts of 2048 x 8192, 8192 tokens per PE),
no LB, as the runtime and the app were fixed during the performance work:

| state                                                        | ms/step |
|--------------------------------------------------------------|--------:|
| first version (one stream per PE)                            |     275 |
| runtime: cross-process receives no longer host-synchronized  |     335 |
| app: comm stream + 4 compute lanes                           |     239 |
| app: sends tagged with dedicated data-ready streams          |     206 |
| app: gather on the send stream                               |     198 |

The max-PE GPU time is 153 ms throughout, so the step is now 77% compute.
What was found, in order (runtime changes are in `src/`, see git):

- **Every cross-process device receive was host-synchronized.** The receive
  path's "same process, different device" case compared node-local device
  indices, which differ for every peer process too; with one device per
  process every direct IPC receive waited on the host for the sender's stream
  and then drained its own. Measured as ~3 ms per 1 MB receive, an exchange
  at 0.5 GB/s on 24 GB/s links, and no overlap of transfers with compute.
  Fixed in `ckrdmadevice.C`; pic2d's no-LB run went from 1.65 s to 0.72 s
  with the same checksum.
- **A send is tagged with its stream's tail at send time.** With the
  host waits gone, a landing copy carries a device-side wait on the sender's
  stream. Anything queued on the sender's stream after the data was produced
  (another expert's GEMMs on a shared lane, landings waiting on other PEs on
  a shared comm stream) delays the receiver's stream and everything behind
  it. Hence three stream roles per PE: `comm` for every landing and the
  combine, compute lanes for the experts, and one send stream per sender
  that waits only on the event marking its own data ready.
- **Runtime hygiene found on the way:** plain DEVICE-mode pups did not note
  pool sources as read (a pool free could race the pack copy); the unpack
  copied on the legacy default stream behind every other payload's landing;
  the polled event queue was one FIFO across streams, so a landing recorded
  after a 150 ms compute-done event was not noticed until the kernel ended.
  All three are fixed; none was the bottleneck here.

What is left, from the trace: token exchange 33 ms, output exchange 10 ms,
combine and drain 7 ms per step. Consumption of the 48 remote slabs a PE
receives is paced at 0.3 to 0.5 ms per slab whatever their size (26 MB and
96 MB take the same 30 ms), and the profile is flat across the scheduler,
network progress, pxshm and event queries: per-message host work in the
device zerocopy path, roughly 0.35 ms all-in on the receiver. Fewer, larger
messages (a slab per destination PE instead of per expert) would remove most
of it at the cost of location transparency; a cheaper per-message path in
the runtime would keep it.

Load balancing, final build (DiffusionLB, `-f 5 -b 5`, async `-l 3`), after
the source fusion and the load-model fix below:

| config                      | no LB | sync | async |
|-----------------------------|------:|-----:|------:|
| defaults, 30 steps, ms/step |   191 |  199 |   185 |
| 4096 x 14336, 20 steps      |   560 |  545 |   530 |

Before those two changes the same runs were 205 / 216 / 207 and 631 / 623 /
605. What changed:

- **One chunked pass per expert, not one per source** (the default; `-U`
  restores the old way). A chunk runs two weight-update GEMMs that read and
  rewrite both weight matrices whatever the chunk holds, so a step's cost is
  set by the number of chunks, and per source that is at least one per
  (source, expert) pair: a typical expert receives a few hundred tokens from
  each dispatcher and paid a full weight update for each. At the defaults that
  is 260 chunked passes per step against 78, and 30 ms per GPU of weight
  traffic. The expert's tokens are copied into one run first and the outputs
  copied back, about 0.8 ms per GPU, so the sends and both pup phases are
  unchanged. Concatenating in dispatcher order keeps the chunk boundaries a
  function of the routing alone: the checksums differ from the unfused ones in
  value and are still identical across placements. Compute span 154 -> 131 ms.
- **Seconds per token is pooled per PE, not per expert.** An expert timed
  itself with events around its own kernels, but up to `-n` experts share the
  device, so the span was inflated by however much company it had -- and a hot
  expert, outliving its lane-mates, had less of it. Per-expert figures
  compressed exactly the skew the balancer has to see. Every expert here does
  identical work per token, so one pooled figure is both the right model and an
  accurate one. `CHARM_MOE_FLATLOAD=1` declares an exact token-proportional
  load instead, which separates a balancer that cannot use a good signal from
  a bad signal.
- **The checksum's copy is waited for before a pack reads it.** `h_chk` is
  filled by an asynchronous device-to-host copy, and `pup_device_order` orders
  only DEVICE-mode copies, so a migration in that window packed whatever was
  there before. It cost one expert's slot in the `w2` reduction, seen once in
  eight async runs at Llama size as a deficit of exactly 1/64. The outputs
  were never affected: `out2` matched in the failing run.

What the balancer still leaves on the table: between reshuffles DiffusionLB
holds `tokens/PE max/avg` at 1.20 to 1.33, where a greedy re-placement reaches
1.01 (see `ref/`). The per-step move cap is a count applied to objects of very
unequal load, so it can stop a shed part-way -- a node asked to shed 0.187 of
its load stopped after four objects with 0.102 unshed and the job sat at 1.9x
for five steps. Lifting the cap (`+LBDiffusionMaxMoveFrac 1`) does recover in
one step and holds 1.2 to 1.33, but moves 35 to 54 objects where the capped
runs move 6 to 9, and the step time does not improve. At this object size the
migration traffic pays for the balance it buys; a cap expressed in the load
dimension rather than by object count is the open item.

Balancing cuts tokens per PE from 1.9x to 1.3x and the GPU-time imbalance
from 1.5x to 1.15x. In the Llama-sized run the balanced steps take 476 to 497
ms against 575 to 601 ms without balancing, until the hot set reshuffles and
the next LB step catches up; the averages above include the reshuffle steps.
Async now overlaps everything: its first LB step costs nothing visible (228
ms against 529 ms in sync at the defaults).

Tracing: `CHARM_MOE_TRACE=1` prints timestamped LB, migration and step
events per PE, `=2` adds each expert's compute and send events, `=3` each
slab's post and consume. `ranklogs/phases.py <dir>` decomposes a level-2
trace into routing, token exchange, compute, output exchange and drain;
`ranklogs/tracesum.py` summarizes migration costs; `ranklogs/pairs.py`
per-pair latencies from a level-3 trace. `CHARM_MOE_HOSTSYNC=1` restores the
original host-synchronized pack and free for A/B runs. With more than one
lane the per-expert device time is a wall-clock span on a shared device, so
`gpu/PE max` overcounts; its ratio across PEs is still meaningful.

Running a single PE: a one-task `srun` step owns only eight of the
allocation's cores, so the NUMA wrapper's `+pemap` aborts the run
(`CmiSetCPUAffinity failed`) with an empty log. Launch single-PE tests
without the wrapper.

## Reference implementation

`ref/moe_ref.py` is the same layer in PyTorch with NCCL, written the way an
MoE training framework writes it: one `all_to_all_single` per exchange instead
of one message per expert, one chunked pass per expert instead of one per
source, and expert re-placement between steps instead of object migration. Its
routing, initialization and update are ports of this example's, checked bit for
bit by `ref/selftest.py`, so the two produce the same checksums and the same
imbalance and can be compared step for step on the same node.

On four A40s at the defaults the example takes 204 ms per step without
balancing against the reference's 189, and 195 ms balanced against the
reference's 153. The gap is not the migration mechanism: the example's async
balancing beats its own sync, while the reference's does not, because
overlapping NCCL weight transfers costs a step of stale placement and shares
the links with the dispatch. The gap is the placement decision, where greedy
re-placement reaches a token imbalance of 1.01 against DiffusionLB's 1.28, and
the GEMM structure, where one pass per expert is worth 30% of the compute
time. `ref/README.md` has the tables and the derivations.
