# Load balancing without flags (2026-10-18)

The runtime chooses the balancing strategy. A run script passes no `+balancer`, `+LBCommOn`, `+LBDiffusion*`,
`+LBAsync`, `+LBCostConfig` or `+LBMetis*`; if a result ever needs one of them, the runtime's choice is the defect.

## What the runtime decides

| Decision | Default | Where | Override (experiments only) |
| --- | --- | --- | --- |
| Strategy when no `+balancer` is given | MetisLB once, then DiffusionLB at every later step (when both are linked: `-module EveryLB -module DiffusionLB`) | `LBMgrInit` (LBManager.C) | `+balancer X` |
| Transfer-cost table | built in: the lbcalib table measured on Delta (A40 x4 per host, Slingshot, 2026-09-15); every balancer prices communication and migration | `DiffusionCostConfig::loadDefault` | `+LBCostConfig <file>` (a file that loads replaces the table; one that fails falls back) |
| MetisLB accepts a partition when | the PRICED step bound (load + communication at its tier + migration amortised over the placement's lifetime) falls by the floor | MetisLB.C verdict | the raw cross-group cut ceiling (`+LBMetisMaxCutRise`) now guards only the uncalibrated case |
| Imbalance floor (Diffusion and Metis) | 3% of the mean (was 10%, which hid a 10-20% drift on sph3d; moves are priced, so a low floor does not churn) | LBManager.h `_lb_diffMinImbalance` | `+LBDiffusionMinImbalance`, `+LBMetisMinGain` |
| DiffusionLB object metric | communication graph (was centroid, which needs application positions) | LBManager.C | `+LBDiffusionCentroid` |
| MetisLB host weights | message counts scaled to the measured host time (a PE's host time tracks the messages it handles, r = 0.82 on sph3d) | MetisLB.C | `+LBMetisHostByTime` |
| Load dimension | whichever binds (host, device or launch), both when comparable (`LBLoadDim.h`) | unchanged | `+LBDiffusionGpuDim`, `+LBDiffusionHostDim` |
| Asynchronous balancing | on when the application asks (`CkLBSetAsync(true)` from its main chare, driven by its own `-a` / `-lbasync`) | `LBManager::setAsyncMode`, broadcast; a request before the group exists is applied when `LBMgrInit` creates it | `+LBAsync` still works and then wins |

The decision prints one line each at startup (`CharmLB> default strategy ...`, `CharmLB> transfer-cost model ...`,
`CharmLB> asynchronous balancing on by the application`) and one line per MetisLB decision (`MetisLB priced: ...
applied | below the floor`), so a log shows what was chosen without `+LBDebug`.

## What the applications still carry

These are step-structure parameters of each application, not balancer flags: how often to call AtSync (`-f`,
`-b`, or leanmd's positional period), whether to overlap the balancing step (`-a` / `-lbasync`) and the lag
(`-l` / `-lblag`). The measured best values are the applications' own defaults now, so a bare command line
balances: sph3d every 25 steps, lag 10, async; sph2d every 2000, lag 1800, async; pic2d every 50, lag 40, async;
leanmd lag 16, async (its period is a positional argument). `-f 99999` turns balancing off; `-A` (leanmd:
`-lbsync`) runs the unsplit barrier. sph3d sorts its particles every 100 steps by default (the GPU work grew 21%
over 2000 steps without it). A period chosen from the measured step time instead of a step count is the natural
next step and is not done.

## Validation

Job 22990411 (2 nodes, gpub[080,087], 16:36-17:15), every arm with no balancer flags. ms/step (leanmd: steps
42-100; sph3d, sph2d, pic2d: whole-run average), best hand-tuned value in the last column:

| Application, configuration | No balancing | Runtime's choice | Best hand-tuned |
| --- | --- | --- | --- |
| leanmd 16x8x8 async, 2 nodes | 1495 (earlier) | 848.1 (821.4 with DiffusionLB's floor at 3%) | 844.3 |
| leanmd 16x8x8 sync | | 856.9 | 872.3 |
| leanmd 8x4x4 async | 202 (earlier) | 116.9 | 134.3 |
| sph3d 98M, 8 A40s, 2000 steps, async every 25 | 56.9 | 38.1, 38.2 (Metis re-evaluated every 4th event); 38.1-41.7 with Metis once | 36.9-38.9 |
| sph2d moving load, 60000 steps, async every 2000 | 2.92 | 1.66 (1.86 with the floor at 3%; 2.22 with the device dimension forced) | 1.62-1.68 |
| pic2d 300 iterations, async every 50 | 27.8 | 27.7 (MetisLB priced the partition at -0.3% and refused; nothing moved) | no gain known for this recipe |

What the runs decided and what was changed on the way:
- MetisLB applied leanmd's partition on priced step time (bound 24.6 to 13.7 s per interval, a 44% fall, while
  the raw cut rose 67%); the hand-tuned runs needed `+LBMetisMaxCutRise 2.0` for the same mapping.
- The load dimension chosen by criticality was right where it was tested: device for leanmd and sph3d, launch
  for sph2d (forcing device there cost 20%).
- DiffusionLB's floor cannot serve both: 3% gave leanmd 821 and sph2d 1.86; 10% gives leanmd 848 and sph2d
  1.66. It stays at 10%, and MetisLB has its own priced floor of 3%. Pricing DiffusionLB's moves well enough to
  need no floor is the open item that would recover leanmd's 3%.
- sph3d's slow drift is not followed by DiffusionLB (nothing moved across nodes at 67 of 79 events), so the
  default chain re-evaluates MetisLB every fourth event: 17 of 20 priced decisions applied a new mapping (bound
  falls of 3-11%, 144-200 of 256 patches each) and the step came down from 40.2 (mean of four Metis-once runs)
  to 38.1, inside the hand-tuned band.
- `-f 99999` (balancing off) tripped the lag check once the applications defaulted to async; fixed.

## Open

- The MetisLB re-evaluation is every fourth event, a fixed cadence. A request from DiffusionLB when the binding
  dimension's imbalance stays above its floor would make it adaptive.
- DiffusionLB's 10% floor costs leanmd about 3%; a priced per-move decision would remove the floor.
- The built-in table is Delta's. Online calibration from the application's own receive times per tier (the
  `[zc-time]` statistics exist) would remove the machine dependence.
