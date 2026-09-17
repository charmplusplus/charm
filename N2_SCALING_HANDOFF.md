# Why load balancing stops paying at two nodes

> **Targeted policy implemented; GPU testing pending:** see
> [DIFFUSION_RESEED_HANDOFF.md](DIFFUSION_RESEED_HANDOFF.md) for the new
> locality-preserving reseeding rule, passing standalone tests, build status,
> and old/new/grow-any plus stencil comparisons to run next.

> **Confirmed placement restriction, job 22132598:** see
> [N2_GROW_ANY_RESULTS.md](N2_GROW_ANY_RESULTS.md). A matched two-node sync
> comparison with `CHARM_DIFFUSION_GROW_ANY=1` improved mean step time from
> 1530.5 to 1391.8 ms and GPU maximum/average from 1.395 to 1.242, with the
> calibrated cost checks retained. Default selection stops after one move
> on overloaded donors with no cost/capacity rejections; the local frontier
> filter prevents further candidates. This is an experimental override, not
> a production policy change.

> **Follow-up, job 22131761:** see [N2_JOIN_RESULTS.md](N2_JOIN_RESULTS.md).
> The two-node force-join experiment found persistent GPU imbalance: in the
> last measured sync interval, GPU work ranges from 8.54 to 22.64 seconds,
> with maximum/mean 1.40. Long force joins are observed, but the claims below
> that GPU work is equalized and only an unavoidable fan-in ceiling remains
> are not established. Each cell waits for 27 force returns, not 14.

Branch `rate-aware-gpu-lb`, Delta A40, 2026-09-16. Everything below is measured
unless it says otherwise. Paths are relative to the repository root.

## The question

LeanMD weak scaling holds 128 cells/GPU constant. Balancing wins 30% at one
node and almost nothing at two, and this is **not** an async-vs-sync question --
the sync arm loses the win too.

| leanmd weak | noLB | sync | removed by balancing |
| --- | --- | --- | --- |
| N=1 | 1419.5 ms/step | 990.1 | **429 ms** |
| N=2 | 1576.0 ms/step | 1517.2 | **59 ms** |

Device work per GPU is ~800 ms/step in all four cells. The balancer equalises
device time and nothing else.

## What is established

**1. The step has ~740 ms/step of non-device critical path at N=2, ~190 ms at
N=1.** Per 20-step LB interval, from the `+LBDebug 1` lines:

| round | N=1 device / interval | N=2 device / interval |
| --- | --- | --- |
| 1 | 14.78 / 26.84 s | 15.05 / 30.50 s |
| 2 | 15.66 / 18.86 s | 15.91 / 33.90 s |
| 3 | 15.83 / 19.45 s | 16.07 / 28.19 s |
| 4 | 15.96 / 20.72 s | 16.19 / 28.98 s |

Device work per GPU is the same at both node counts, as weak scaling intends.
The interval is 45% longer at N=2. Device share of the step is 77-83% at N=1
after the first round and 47-57% at N=2.

    grep -aE "load dimension:|measured loads explain" \
      examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N{1,2}_sync/rank_0.log

DiffusionLB prints its own verdict at N=2, from the first round, in both sync
and async: `NO dimension holds the step open; balancing any cannot move it much`.

**2. That idle absorbs the device imbalance rather than adding to it.** This is
the part that answers "why is sync not better either". A process already waiting
~740 ms on something else is not slowed by also holding 400 ms of surplus device
work -- the surplus overlaps the wait. So removing the imbalance shortens the
step by 59 ms instead of 429 ms. noLB and sync converge at N=2 not because the
balancer failed but because what it fixes stopped being on the critical path.

**3. Receive volume is identical at both node counts.** 707,840 device receives
per process over 100 steps (7,078/step), `MEMCPY=586880, IPC+RDMA=120960`, in
the noLB arm at N=1 and N=2 alike. Volume is not what changes.

    grep -ah "zc-stats" examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N{1,2}_nolb/rank_*.log

**4. The only structural difference at N=2 is the transport mix, and it splits
the processes in half.** Four of the eight processes send 60,480 of their
cross-process receives over RDMA; the other four report `OTHER=0` and never
touch the fabric. `findTransferModeDevice` (`src/ck-core/ckrdmadevice.C:61`)
returns RDMA only between physical nodes, so at N=1 the tier never runs at all.

**5. Receive outstanding-time tracks receiver LOAD, not transport.** In the N=2
noLB arm, sorted by the process's share of the density gradient:

| pid | mean bytes | MEMCPY mean | IPC mean |
| --- | --- | --- | --- |
| 1460828 | 15,020 | 10.4 ms | 6.6 ms |
| 1460829 | 21,898 | 28.4 ms | 15.4 ms |
| 1460830 | 28,800 | 42.5 ms | 47.2 ms |
| 1460831 | 35,678 | 61.6 ms | 99.3 ms |
| 3732598 | 42,580 | 84.9 ms | 23.2 ms |
| 3732599 | 49,458 | 127.2 ms | 126.5 ms |
| 3732600 | 56,359 | 289.5 ms | 583.3 ms |
| 3732601 | 63,252 | 350.5 ms | 439.1 ms |

A same-process MEMCPY receive contains no network at all, yet on the heavy
process it averages 350 ms, and it rises with load exactly as IPC does. On
pid 3732599 the two transports are 127.2 vs 126.5 ms -- 0.6% apart.

    grep -ah "zc-time" examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_nolb/rank_*.log | sort

**Caution for whoever reads these numbers next:** `mean_us` is time-outstanding
with heavy overlap, not serial cost. At N=1 sync, MEMCPY receives average
95-238 ms against a 990 ms step, so roughly a thousand receives are in flight at
once. Do not multiply `mean_us` by the receive count and compare it to the step.
An earlier pass of this investigation did exactly that and produced a bogus
"~900 us per RDMA receive" figure.

## What is ruled out

- **Bandwidth.** ~84 MB/step crosses the fabric; at the measured `inter_node`
  beta (5.42e-11 s/byte = 18.5 GB/s) that is 4.5 ms of the 740.
- **Per-message fabric latency.** ~605 cross-node receives per process per step
  at the measured alpha (5.195e-05 s) is 30 ms even if perfectly serialised.
- **The migration repair paths.** Counted directly, whole 100-step run, eight
  processes: restage fired **0** times; forward repair fired **190** times
  (116, 69, 4, 1, 0, 0, 0, 0). Roughly 2 ms/step across all processes. The
  restage path is dead under `+gpupool`.
- **A migration cap.** `_lb_diffMaxMoveFrac` defaults to 1.0 (`LBManager.h:118`)
  and the budget code at `DiffusionCore.C:183` only engages below 1.0.
  `+LBPercentMovesAllowed` is GreedyRefine's and is not in the Metis->Diffusion
  chain. N=2 migrates *more* than N=1 (`1472 509 866 3` vs `1187 0 0 0`).
- **Balancer failure on its own objective.** `Max/Avg load per PE AFTER LB`
  reads 3.37/2.20 at N=2 and looks like a 61% residual imbalance. It is not
  evidence: `pe_load` is filled from `diffusionObjCpuLoad`
  (`src/ck-ldb/DiffusionHelper.C:132`), so that line reports the **host**
  dimension, which the balancer deliberately did not optimise (alpha_h 0.04 vs
  alpha_g 1.00). Wrong quantity to judge it by.

## The transport hypothesis: tested and DEAD (job 22127364)

The discriminator was within a single N=2 process: it performs both RDMA and
local receives, same load, same steps, so only the transport differs. The
cross-node tier is not slower. It is 100-550x **faster**, while carrying more
bytes per receive.

| pid | MEMCPY mean | IPC mean | **RDMA mean** | RDMA mean bytes |
| --- | --- | --- | --- | --- |
| 1945285 | 34,036 us | 10,307 us | **1,098 us** | 53,747 |
| 1945289 | 66,974 | 106,549 | **443** | 39,312 |
| 42350 | 122,542 | 100,695 | **639** | 39,321 |
| 42354 | 337,416 | 308,526 | **555** | 52,451 |

    grep -ah "zc-time\|zc-stats" \
      examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_sync/rank_*.log | sort -t= -k2

The reason is structural, and it retrospectively explains why MEMCPY and IPC
tracked each other so closely in the noLB table above. An rget is a **pull**: the
receiver issues it when ready and the clock measures actual network time. A
MEMCPY or IPC receive is posted and then waits for the receiver's stream to
execute the copy behind all of that process's queued GPU work -- so those
columns were never measuring transport, they were measuring how far ahead of the
local GPU the receive was posted. Cross-node communication is not the bottleneck
at N=2; it is the cheapest thing in the step.

Results of that run: noLB 1581.0, sync 1528.8, async 1472.0 ms/step. All three
arms rc=0 -- the first exercise of the async repair park-and-resume path under
real cross-node migration (`migr 1502 489 895 3`), so the abort described below
is fixed.

## Unattributed GPU work: tested and DEAD (jobs 22128402 / 22128403)

Both halves fail.

**Kernel attribution is ~100%.** At both node counts, every drain:

    kernels=217216 ... attributed=217216 unattributed=0   deferred=0
    kernels=297048 ... attributed=296792 unattributed=256 deferred=0

Nothing is dropped in correlation, nothing deferred, no unresolved tokens. The
balancer sees every kernel CUPTI traces.

**Device-to-device copies are ~2% of the interval.** They were never traced at
all (`CUPTI_ACTIVITY_KIND_MEMCPY` is not enabled by default); now they are,
behind `CHARM_CUPTI_MEMCPY`. Differencing the cumulative counters across the
last two drains on process 0, N=2 sync, interval 29.379 s:

| counter | delta | |
| --- | --- | --- |
| `kernel_s` | +26.17 s | raw sum across concurrent streams |
| `memcpy_s` | +0.58 s | **1.97% of the interval** |
| `memcpy_n` | +339,820 | ~17,000 copies/step/process |

17,000 copies a step, 29 ms of device time. Numerous and tiny.

So `host 3% + device 51% + launch 12% + memcpy 2%` leaves about a third of the
step unaccounted, and the GPU is **not** secretly busy during it. The idle is
real.

**Two traps in reading these counters.** They are file-scope per *process*,
shared by its 8 PEs, so the `pe=` label is whichever PE ran the drain, not a
per-PE breakdown. And `kernel_s` is a raw sum over concurrent streams -- 26.17 s
against a 29.4 s interval is NOT 89% occupancy. `T_g` (15.09 s) is the
concurrency-corrected figure, SM-weighted via `computeKernelSMs`. Do not compare
`kernel_s` to `T_g` directly.

**Build trap found here:** `make ck` does **not** rebuild `hapi_impl.cpp` --
it lives in `libhybridapi.a`, which has its own `make hybridapi` target. An
instrumented run silently produced no new counters because of this. Verified
that no earlier result was affected: the HAPI changes committed in `8b402a766`
are present in the previous library (tested by two distinctive string literals
from the commit diff).

## The open question

One candidate survives: **the per-step fan-in.** Each cell has 14 computes
(14,336 / 1,024) and cannot integrate until all 14 return forces, so its step is
set by its slowest contributor. At N=2 those span twice as many processes, so
the step waits on the max of more independent stragglers. It predicts what is
observed -- idle GPU, no dimension holding the step open, and identical
behaviour in noLB, sync and async.

**Test:** time the cell-side join directly. Per cell per step, record the
interval between the first and last force arrival, and which process the last
one came from. Not a transport measurement and not a device counter -- both of
those are now exhausted.

If it holds, the N=2 ceiling is structural: no placement of objects fixes a
fan-in tail, and the conclusion is to stop expecting this balancer to recover
it. What would help instead is changing the dependency structure -- fewer, larger
contributions per cell, or overlapping the join with the next step's work.

Not chased, deliberately: per-process host load swings 2.3-25.6 s between
processes and rounds (`grep -a "node host: sum=" weak_N2_sync/rank_*.log`), but
host is 3-4% of the step by the balancer's own measure, so it cannot be the
740 ms.

## Instrumentation gap that was closed

The RDMA slot printed no `[zc-time]` row at all while its mode counter showed
60,480 receives per process. The per-message tally in `DeviceRdmaInfo` only
records once **every** op of a receive has completed
(`src/ck-core/ckrdmadevice.C`, the `info->counter == info->n_ops` block); an
rget issued alongside an op that resolves some other way never reaches it.

Fix: `DeviceRdmaOp::rget_posted` (`src/util/cmirdmautils.h`), stamped where the
rget is issued and read where it completes, so a cross-node receive cannot be
lost to the aggregation. `rdma_data` comes from `CmiAlloc` and is not zeroed, so
the field is explicitly cleared at op setup -- without that, stale storage reads
as a timestamp and charges nonsense to the tier.

## Code in the tree

Committed earlier on `rate-aware-gpu-lb`:

- `8b402a766` async LB: keep an element's measurement gate closed across
  migration. See `ASYNC_LB_MEASUREMENT_FIX.md`.
- `3d4ad3203` lbcalib inter_node tier measured on two nodes.
- `729ca3f54`, `b6d7201ee` application/LB-instrumentation cleanup.

Uncommitted (`git diff --stat`: 6 files, +256/-48):

| file | change |
| --- | --- |
| `src/ck-core/ckrdmadevice.C/.h` | inter-node restage put and forward repair no longer block the PE; three-way `CkDeviceRepairResult`; per-op rget timing |
| `src/ck-core/ckarray.C` | caller updated for the three-way result |
| `src/util/cmirdmautils.h` | `DeviceRdmaOp::rget_posted` |
| `examples/.../sph2d/sph2d.C` | async lag clamp validates against the real trigger gap |
| `examples/.../gpudirect/scale.sh` | sph2d lags 800/200; budget check sized from the expectation |

### A bug introduced and fixed here -- read before trusting the async arm

Making the repair paths asynchronous changed `CkRdmaDeviceRepairForward`'s
return from "redirected" to "consumed (redirected **or** parked)".
`device_forward_redirect_bridge` treats a second redirect as fatal, so every
parked message aborted:

    Reason: [53] device forward redirect bounced again: the source process
            could not repair its own payload

Seen in `leanmd/ranklogs/weak_N2_async/rank_6.log` (job 22125260, rc=134, after
two LB rounds). Fixed by replacing the bool with
`CkDeviceRepairResult{CallerDelivers, Redirected, Parked}` so the two cannot be
collapsed again. **Validated** by job 22127364: all three arms rc=0 with
cross-node migration at every round.

Note also that both repair paths are rare (0 and 190 firings per run), so
whether they block or not cannot explain the 740 ms. They were made async on
their own merits.

## Log inventory

Summaries, `examples/charm++/cuda/gpudirect/`:

| file | what |
| --- | --- |
| `scale-22119715.out` | N=1 weak, leanmd + sph2d, first run on the fixed gate |
| `scale-22120057.out` | N=2 weak, leanmd + sph2d (rank logs since overwritten) |
| `scale-22120625.out` | N=1 sph2d weak, async lag 800 |
| `scale-N2sph-22121684.out` | N=2 sph2d weak, async lag 800 |
| `scale-22125260.out` | N=2 leanmd, `CHARM_ZC_STATS=1`, async aborted |
| `scale-22127363.out` | N=1 leanmd, `CHARM_ZC_STATS=1` |
| `scale-22127364.out` | N=2 leanmd, `CHARM_ZC_STATS=1` -- killed the transport hypothesis |
| `scale-22128402.out` / `scale-22128403.out` | N=1 / N=2, `LBDEBUG=2 CHARM_CUPTI_MEMCPY=1` -- killed the unattributed-GPU hypothesis |

Rank logs. These are `rm -rf`'d per run, so they hold the **most recent** run of
that shape only:

| path | job |
| --- | --- |
| `leanmd/ranklogs/weak_N1_{nolb,sync,async}/rank_*.log` | 22127363 (ZC stats on) |
| `leanmd/ranklogs/weak_N2_{nolb,sync,async}/rank_*.log` | 22127364 (ZC stats on, RDMA timed, all arms clean) |
| `sph2d/WEAK_N1_*.log` | 22120625 (lag 800) |
| `sph2d/WEAK_N2_*.log` | 22121684 (lag 800) |
| `sph2d/F3_*.log` | 2026-09-15 reference (lag 1800) |
| `sph2d/WEAK_N4_*.log` | 2026-09-14, 4 nodes (lag 1800) |

What to grep in a leanmd rank log:

    "load dimension:"            T_h / T_g / T_l and the alphas per round
    "measured loads explain"     share of the interval each dimension covers
    "Max load per PE AFTER LB"   HOST dimension -- see the caution above
    "cross node migrations"      moves per round
    "\[zc-stats\]"               receive counts by transport, per process
    "\[zc-time\]"                receives, bytes, total_us, mean_us per transport
    "\[ipc-stats\]"              includes forward_repairs
    "\[PELOAD\]"                 per-PE host load at each LB step
    "cupti-device-total"         kernel_s / memcpy_n / memcpy_s / memcpy_GB
    "hapiProcessCuptiBuffers"    kernels / attributed / unattributed / deferred

The last two need `LBDEBUG=2` and, for the memcpy fields, `CHARM_CUPTI_MEMCPY=1`.

`[zc-stats]`, `[zc-time]` and `[ipc-stats]` only appear when `CHARM_ZC_STATS` is
set. `ZC RESTAGE` lines need `CHARM_ZC_RESTAGE_DEBUG`.

## Reproducing

    cd examples/charm++/cuda/gpudirect
    export CHARM_ZC_STATS=1
    PART=gpuA40x4-interactive bash submit_scale.sh "1 2" leanmd weak "nolb sync async"

One job per node count, each sized to that count. `DRY=1` prints the plan
without submitting. The runs use `+gpupool` with a pre-sized arena (2048 MB at
N>=2 -- 4096 aborts the fabric), one process per GPU, and the NUMA wrappers;
`scale.sh` carries the reasoning for each.

## A separate finding, already fixed

sph2d's async and sync arms were not doing equal work and had not been since
2026-09-14. `sph2d.C` drops any LB trigger landing inside an open lag window
(`isLBIter(it) && !lb_waiting`). Weak sph2d ran `-f 1000 -b 2000 -l 1800`: first
LB at 1000, period 2000, so triggers are 1000, 2000, 4000, 6000 and the shortest
gap is **1000, not 2000**. The 1800-step window swallowed the 2000 trigger, so
sync balanced four times and async twice. The app's guard checked the lag
against the *period*, which the offset first-LB step makes the wrong quantity.

With `-l 800`: N=1 14.662 both arms (was 14.588 sync / 14.674 async); N=2 sync
15.279, async 15.177 (was 15.464 / 15.571). `sph2d.C` now computes the true
minimum trigger gap and aborts on a lag that straddles one. Compare
`sph2d/F3_*.log` (lag 1800) against `sph2d/WEAK_N1_*.log` (lag 800) via
`grep -a "LB at step"`.
