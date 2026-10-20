# leanmd at 2 nodes: what is left after load balancing

Written 2026-09-17. Everything below is from jobs run that day on Delta A40,
2 physical nodes, 4 processes/GPUs per node, 8 PEs per process, weak grid
`16 8 8` (1024 cells, 128 cells/GPU), 100 steps, first LB at 20, period 20,
`-computemap local -density gradient`, MetisLB then DiffusionLB, `+gpupool`
2048 MB arenas, NUMA-pinned via `leanmd/ranklogs/rankwrap.sh`.

The reference point is the same benchmark at 1 node (job 22119715): noLB 1393.8,
sync 993.5, async 987.1 ms/step.

## The headline

At 2 nodes the balancer's remaining prize is small and the step is often not
GPU-bound at all. Two sync runs with identical configuration:

| sync arm, same config | job 22154226 | job 22155894 |
| --- | ---: | ---: |
| step time | 1257.6 ms | 1388.9 ms |
| final measured GPU imbalance | 1.113 | 1.071 |
| busiest GPU, per 20-step interval | 24.4 s | 23.5 s |
| total GPU seconds, 8 procs, whole run | 810.1 | 810.7 |
| last-interval wall (steps 81-100) | 26.37 s | 28.52 s |

The slower run is the better balanced one, with less work on its busiest GPU and
the same total GPU work. **Placement is not the variable**; the 131 ms/step is
time not accounted for by the balance of GPU work.

**Do not read `cupti-device-total kernel_s` as GPU occupancy.** It sums kernel
durations, and leanmd runs concurrent kernels on several streams per process, so
the sum can exceed the wall interval -- job 22156615's sync arm gives 24.2 s of
kernel time in a 23.02 s interval. An earlier version of this file drew an
"occupancy" number from that ratio and concluded the GPUs were idling a fixed
fraction of each interval; that inference is not supported. What the metric does
support is a comparison BETWEEN runs: total kernel work is constant
(810.1 / 810.7 / 812.7 s over 8 processes) and the imbalance sits in a narrow band
(1.07-1.11), while step time ranges over 1156.6-1388.9 ms.

## Ruled out, with the evidence that rules it out

1. **Peer decision / home-forwarding is NOT the cause of async's per-step cost.**
   `+LBAsync` forces `+LBPeerDecision` (`src/ck-ldb/LBManager.C:435`), which drops
   the post-migration `BroadcastLocationUpdate` (`src/ck-ldb/DiffusionLB.C:1114`,
   `src/ck-ldb/DistBaseLB.C:328`), and the comment above the forcing records a
   measured 14% penalty on ordinary iterations in barnes. async's ordinary steps
   were 1313 ms, sync's 1148 -- a 14.4% match. It is a coincidence. Running with
   `CHARM_LB_ASYNC_NO_PEER_DECISION=1` (job 22155894) left async's ordinary steps
   at **1308 ms**, unchanged from 1313. What moved was sync, 1148 -> 1329, and the
   env var does not touch sync's code path. Energy conservation passed with the
   forcing disabled.

2. **Placement quality is NOT the cause of the run-to-run variance.** See the
   table above: the slower sync run has the better imbalance (1.071 vs 1.113).

3. **Migration volume is NOT what changed sph2d.** sph2d's across-node phase moved
   **2 objects** in a whole sync run and **4** in an async run (per-node counts over
   all LB steps). No budget or plan change has anything to act on there.

## Established this session

- **DiffusionLB's decision floor was stopping the plan on a gradient.** The floor
  (`+LBDiffusionMinImbalance`, 0.10) was held against the *neighbourhood* mean, and
  a ramp around the ring backbone has a per-hop slope under the floor while the job
  spans far more. leanmd node 6 at LB step 2 of job 22137894: load 20.880676,
  neighbours 17.432060 and 18.175282, threshold 1.780367, overload 1.724308 --
  refused, with a declared spread of 13.40..21.16. Fixed by holding the floor
  against the job-wide mean as well; measured GPU imbalance then went
  1.79 -> 1.149 -> 1.143 -> 1.142 against a 1.25 plateau (job 22151249).
  The floor is applied in **three** places in one round -- the node's overload, the
  per-edge send, and the flow after momentum -- and all three needed the exception.

- **MetisLB was refusing a mapping worth ~200 ms/step.** Its own pricing:
  `step bound 24.55 -> 15.96, comm and migration included (saves 8.584 s/interval,
  35.0%); cross-group cut 10.190 -> 15.512 (+52.2% against a ceiling of +0.0%);
  3329 migrations 0.054 s/interval; raises the cross-group cut, current mapping kept`.
  With `+LBMetisMaxCutRise 1.0` the mapping applies and sync ran 1257.6 (job 22154226)
  and 1388.9 (job 22155894) against 1455.6 without it. A win on the mean, but the
  two admitting runs differ by 131 ms, so it is not sized by these runs.

- **The declared loads mis-rank the processes.** Declared vs measured GPU seconds,
  job 22151249 final interval: ratios 1.25, 1.54, 1.45, 1.37, 1.33, 1.32, 1.30, 1.32
  for nodes 0..7. Not a constant factor, so balancing declared load to convergence
  does not balance measured GPU work, and the balancer's objective is distorted by
  up to 23% between processes.

- **The comm graph was over-counting under async, and the fix is visible in the
  balancer's input.** `LBDatabase::Send`/`MulticastSend` recorded an object's traffic
  regardless of its `joinedStep` state, while its compute measurement was gated on it;
  under async an object keeps sending after its window closes, so its traffic was
  billed to an interval its compute was not. With the early return added
  (`LBDatabase.C`), the async arm's pre-LB cross-group cut became **10.190485**,
  exactly sync's figure, where before the fix it read 10.259108 -- the sync/async
  asymmetry in the balancer's input is gone, and async ran 1440.0 -> 1351.2.

  **The fix is async-only, and the logs prove it.** The gate is per object and only
  bites where an object keeps executing after its window closes; under sync the
  element parks at `ResumeFromSync` and is never the running object. Sync's pre-LB
  cross-group cut reads **10.190485 both before and after** the fix, identical to
  7 significant figures, so its recorded comm graph did not change. Sync's 1257.6 ->
  1156.6 in the same pair of jobs is therefore run-to-run variance, not the fix --
  the same two-job comparison that gave 1257.6 and 1388.9 for identical configuration
  puts that spread at 131 ms. 1156.6 is the best N=2 sync figure recorded (-27.2%
  against noLB 1588.6, against -28.7% at 1 node) but it is not attributable.

## Numbers, per job

| job | config | noLB | sync | async |
| --- | --- | ---: | ---: | ---: |
| 22135839 | reseed fix, old budget, old floor | 1556.3 | 1442.4 | 1457.5 |
| 22137894 | + relay budget | 1623.6 | 1447.3 | 1509.5 |
| 22151249 | + global-mean floor | 1586.2 | 1455.6 | 1421.0 |
| 22154226 | + `+LBMetisMaxCutRise 1.0` | 1582.9 | 1257.6 | 1440.0 |
| 22155894 | + `CHARM_LB_ASYNC_NO_PEER_DECISION=1` | 1577.7 | 1388.9 | 1307.5 |
| 22156615 | + comm-instrumentation fix (peer decision back on) | 1588.6 | **1156.6** | 1351.2 |

Per-step decomposition (rank_0 `Step N Benchmark Time`, means over the run):

| | LB steps (n%20==0) | step after LB | other 90 steps |
| --- | ---: | ---: | ---: |
| 22154226 sync | 2324 | 2121 | 1186 (min 375, max 3688) |
| 22154226 async | 2279 | 1210 | 1325 |
| 22155894 sync | 1640 | 1269 | 1329 |
| 22155894 async | 1463 | 1500 | 1308 |

Async does overlap what it claims to: whole-run LB overhead 4317 ms against sync's
9429 ms in job 22154226. Note the spread on ordinary steps -- 375 to 3688 ms in one
arm -- which is itself unexplained and is the same idle-time phenomenon.

## Open questions, in the order I would take them

1. **What is the wall time that the GPU work does not account for?** Runs with the
   same total kernel work (810-813 s over 8 processes) and the same imbalance band
   (1.07-1.11) range from 1156.6 to 1388.9 ms/step. Note the caveat above: summed
   kernel seconds cannot be turned into an idle fraction on this application.
   `LEANMD_JOIN_TRACE=1` traces the per-cell force join,
   which is where a cell waits for off-node contributions; one traced run should say
   whether the idle is inbound-data wait and on which ranks. This is the largest
   item on the board and it is not a load-balancing problem.

2. **Why do the declared loads mis-rank processes by up to 23%?** Costs no GPU time;
   the declared-load path and the CUPTI attribution are both readable from the logs
   already on disk.

3. **Is the sph2d async difference the `+LBDebug` I added?** After job 22135839 I
   added `+LBDebug` to `SPH_MD` (commit 81ee8014c) and ran later jobs at `LBDEBUG=2`;
   sph2d async read 15.241 before and 16.705/16.405 after, sync was flat. `scale.sh`
   now has `SPH_LBDEBUG`/`LMD_LBDEBUG` so one arm can run quiet. Untested.

4. **noLB itself loses 13.8% crossing to 2 nodes** (1393.8 -> 1577.7..1623.6). No
   placement decision touches that; it bounds everything above.

## Code state (all uncommitted, branch `rate-aware-gpu-lb`)

- `src/ck-ldb/DiffusionCore.C` -- executor sheds the plan's outflow capped at
  holdings, instead of its excess over the neighbourhood mean. `CHARM_DIFFUSION_SHED=excess`
  restores the old rule.
- `src/ck-ldb/DiffusionFlow.h` -- the floor takes a job-wide mean; a node over
  `mean*(1+floor)` may plan under the local floor, capped at how far over it is.
- `src/ck-ldb/DiffusionLB.{ci,h,C}`, `DiffusionPseudo.C` -- `pseudoLoadContribute` /
  `pseudoLoadVerdict`, one collective before the rounds (it must be before the loop:
  a zero-plan step converges on round 0 and never reaches a per-round collective).
  `CHARM_DIFFUSION_GLOBAL_FLOOR=0` restores the old rule.
- `tests/charm++/load_balancing/lbdriver/ringflow_test.C` + Makefile -- `make check-ringflow`,
  login-node safe, runs the real `diffusionRoundFlows` over leanmd's ring with the
  measured step-2 loads. Local floor 1.310 (4.52 planned), global floor 1.132 (12.62),
  noise plans 0.1% of the job.
- `tests/charm++/load_balancing/lbdriver/lbsim.C` -- mirrors both changes; adds
  `LBSIM_RAMP` (linear weight ramp in x, leanmd's shape) and `LBSIM_GLOBAL_FLOOR`.
  32x32 hot disc over 8 nodes: old floor 3.43/2.00/1.51/1.51/1.51, cut 194, 3 detached
  pieces; global floor 3.43/1.93/1.44/1.33/1.30, cut 184, 1 piece. Flat job plans zero
  moves either way.
- `examples/charm++/cuda/gpudirect/scale.sh` -- `SPH_LBDEBUG`, `LMD_LBDEBUG`, `LMD_EXTRA`.

## Where the evidence is

Rank logs, preserved per job (`scale.sh` overwrites the unsuffixed directory):

```
examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_{sync,async,nolb}_22137894_relay
examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_{sync,async,nolb}_22151249_globalfloor
examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_{sync,async}_22154226_cutrise
examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_{sync,async}_22155894_nopeer
examples/charm++/cuda/gpudirect/scale-<jobid>.out
```

Useful greps: `cupti-device-total kernel_s=` is cumulative per process, so per-interval
GPU work is the difference between consecutive values; `AcrossNodeLB: my_load=` carries
the declared load, the neighbourhood mean and the plan; `^Step N Benchmark Time` is the
per-step time on rank 0; `MetisLB priced:` is the refused-or-applied mapping with its
own cost model.

## Caveats

Every number above is a single run. Two identically configured sync arms differ by
131 ms/step, so any effect smaller than that is not resolved by these jobs, and that
includes the size of the cut-ceiling win. The variance is not noise in the usual sense
-- it is the occupancy phenomenon in question 1, and explaining it is more valuable
than averaging over it.
