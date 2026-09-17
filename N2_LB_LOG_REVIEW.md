# LeanMD two-node LB: review of existing logs

Reviewed 2026-09-16 against `N2_SCALING_HANDOFF.md`, current source, and retained
rank logs. No new jobs or runtime changes made. The pending run's
`scale-22127364.out` was not present at review time.

## What the measurements support

Latest completed noLB/sync runs: N1 job 22127363, N2 job 22125260.
Paths below are relative to `examples/charm++/cuda/gpudirect/`.

| metric | N1 noLB | N1 sync | N2 noLB | N2 sync |
| --- | ---: | ---: | ---: | ---: |
| reported mean, steps 42–100 (ms) | 1419.5 | 990.1 | 1576.0 | 1517.2 |
| total application time (s) | 134.128 | 110.746 | 150.275 | 146.227 |
| mean steps 42–60 (ms) | 1386.0 | 1026.3 | 1511.0 | 1626.1 |
| mean steps 62–80 (ms) | 1400.3 | 947.0 | 1559.2 | 1500.5 |
| mean steps 82–100 (ms) | 1433.8 | 994.7 | 1581.7 | 1444.1 |

The steady-window improvement drops from 30.3% to 3.7%; whole-application
improvement drops from 17.4% to 2.7%. N2 improves later in the run (8.7% in
steps 82–100), so a single mean conceals settling behavior. These block means
omit steps immediately following LB triggers, deliberately separating ongoing
step behavior from the immediate pause; they are not whole-run speedups.

The N2 sync mapping moves 1474, 509, 865, 3 objects across process boundaries;
N1 moves 1183, 0, 0, 0. N2 is still changing substantially at step 80.
LB's reported step durations are only 0.241, 0.085, 0.117, 0.071 seconds at N2;
the poor gain persists between those calls.

## Strongest additional clue: balancing changes the communication workload

Summing `[zc-stats]` over all ranks in each complete 100-step run:

| mode | N1 noLB | N1 sync | N2 noLB | N2 sync |
| --- | ---: | ---: | ---: | ---: |
| MEMCPY | 2,347,520 | 2,225,000 | 4,695,040 | 4,513,400 |
| IPC | 483,840 | 607,543 | 725,760 | 804,286 |
| OTHER (cross-node tier here) | 0 | 0 | 241,920 | 347,885 |

N2 sync has **43.8% more cross-node receives** than N2 noLB. These totals
include migration-related traffic and are not pure steady-state force counts.
Separately, Diffusion's measured external communication rises across rounds:
3354, 5079, 5546, 5988 MB. This is inter-process communication, not exclusively
fabric traffic. N1 likewise increases external communication (1589 to 3285 MB)
but can serve all of it within the physical node.

Thus identical noLB receive counts at N1/N2 do not rule out a communication
penalty caused by LB. Moving computes off the dense region trades GPU work
against position/force communication with fixed cells. At N2 that trade can
cross physical nodes. This is a plausible explanation for lost benefit, not a
causal attribution established by the counters alone.

Metis rejects its candidate at N2: predicted bound 24.541 -> 15.955 seconds,
but cross-group cut +53.5% exceeds its zero-rise ceiling. Diffusion subsequently
moves objects. More moves do not establish that GPU balance was achieved, nor
that the chosen placement optimizes the application's dependency path.

## Corrections to the handoff's interpretation

1. `T_g` is `max(sumDev / gpus, maxObjectDev)` in `src/ck-ldb/LBLoadDim.h`.
   It is an ideal-placement lower bound, **not the observed busiest GPU's
   elapsed work**. Subtracting it from wall time does not measure idle time or
   prove a 740 ms irreducible non-device critical path. Residual imbalance,
   scheduling, overlap, and dependencies remain candidates. The printed
   “NO dimension holds the step open” verdict overinterprets this bound too.
   In async, a shortened measurement window makes comparison against the full
   LB interval additionally misleading unless coverage is normalized.
2. Each cell **creates 14 computes but waits for 27 force returns**: one self
   compute plus 26 pair computes shared with neighbors. See `Cell::createComputes`,
   `NUM_NEIGHBORS` in `defs.h`, and `Cell::run` in `leanmd.ci`.
3. `Cell::run` has local dependency joins, not a global barrier each step.
   The printed step timer is cell `(0,0,0)`'s timer. It includes force collection,
   integration, and periodic atom exchange; whole-application completion is
   measured separately. Doubling the job does not automatically double each
   cell's contributor count or prove a placement-independent fan-in ceiling.
4. Large MEMCPY/IPC outstanding times demonstrate queueing/dependencies in
   addition to transfer time. They cannot assign that time to the receiver's
   load alone: sender readiness and stream/callback progress also matter.
5. Small alpha/beta transfer estimates make simple bulk bandwidth saturation
   less compelling, but do not rule out delayed progress, source readiness,
   contention, or feedback through the force/ack dependency graph.
6. The proposed RDMA-vs-IPC mean comparison is useful but not decisive. The new
   RDMA timestamp starts immediately before `rdmaGet`, after preparation and
   destination registration; it cannot see earlier source/descriptor delays.
   IPC timing is aggregated through the message completion path. Match timer
   boundaries and examine tails before interpreting equality as exoneration.

## What remains unresolved and what to measure next

Existing logs cannot distinguish residual GPU imbalance from extra exposed
communication/progress delay. The strongest working hypothesis is a placement
tradeoff: N2 spreads compute but adds costly dependencies across nodes, with
slower mapping convergence than N1. It is premature to call this unavoidable.

For a discriminating follow-up:

- Record per-GPU summed measured work AND actual busy/idle timelines over the
  same steady windows; report maximum/mean, not only the ideal lower bound.
- Time cell first/last force arrival and compute input readiness, kernel
  completion, and final force-send acknowledgment. Tag physical-node crossings.
  This separates waiting for production, transport, and callback progress.
- Compare ordinary LB with placement constrained within each physical node,
  and with a mapping balanced once then frozen. The first tests the compute vs
  fabric tradeoff (while retaining any node-level imbalance); the second tests
  continued remapping and convergence. Use the same executable and repeated runs.

The retained N2 async rank logs abort at step 61 and must not supply a completed
run average. Earlier job 22120057 completed: noLB 1576.5, sync 1540.2, async
1463.7 ms/step, confirming the broad trend but using an older binary. The N1/N2
latest runs also have different executable timestamps and physical nodes;
exact cross-run deltas are observational, not a controlled transport ablation.
