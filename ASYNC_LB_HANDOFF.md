# Async LB non-convergence on leanmd — data handoff

Question: with `-lbasync -lblag 16`, DiffusionLB keeps migrating objects at every
LB step instead of converging after the first. Sync converges immediately.
Why?

All numbers below are measured on Delta A40, 1 node, 4 processes x 8 PEs,
leanmd `8 8 8 100 20 20 -computemap local -density gradient`, MetisLB then
DiffusionLB (`+LBDiffusionCommOn +LBDiffusionGpuDim`), `+LBDebug 1`.
Metric `mean42` = mean ms/step over steps 42-100.

---

## 1. The core observation

Cross-node (= cross-process, 4 GPUs) migrations per LB step. LB steps fire at
simulation steps 20/40/60/80.

| arm                     | LB1  | LB2 | LB3 | LB4 | mean42 |
|-------------------------|------|-----|-----|-----|--------|
| sync                    | 1188 |   0 |   0 |   0 |  993.9 |
| sync                    | 1188 |   0 |   0 |   0 |  974.7 |
| sync                    | 1177 |   0 |   0 |   0 | 1043.0 |
| async lag 16            | 1187 | 636 | 598 |   2 | 1079.5 |
| async lag 16            | 1185 | 636 |   2 |  14 | 1185.2 |
| async lag 16, cost/6    | 1182 | 625 |   2 | 606 | 1075.5 |
| async lag 8             | 1182 |   0 |   0 |   0 | 1039.4 |
| async lag 4             | 1186 |   0 |   0 |   0 | 1003.7 |
| async lag 4             | 1182 |   0 |   0 |   0 | 1027.8 |
| async lag 4             | 1186 |   0 |   0 |   0 |  987.8 |
| async lag 2             | 1181 |   0 |   0 |   0 | 1011.4 |

Churn appears at lag 16 and is absent at lag <= 8. At lag 4 async ties sync
(1006.4 vs 1003.9, n=3 each; run-to-run spread on this benchmark is 5-10%).

---

## 2. The load the balancer diffuses (T_g = device-time bound)

Printed by `+LBDebug 1` as `load dimension: device by flag (T_h .. T_g .. T_l ..)`.
`T_g = boundDev() = max(sumDev/gpus, maxDev)` — LBLoadDim.h:127.

| arm          | MetisLB (1st LB) | Diff step1 | step2 | step3 | step4 |
|--------------|------------------|-----------|-------|-------|-------|
| sync         | 13.046406        | 14.774451 | 15.675879 | 15.835450 | 15.967522 |
| sync (run 2) | 13.044749        | 14.641188 | 15.599847 | 15.821894 | 15.953537 |
| async lag 4  | 13.049776        | 11.823222 | 13.441687 | 12.669369 | 12.783274 |
| async lag 16 | 13.044784        |  2.946834 |  7.079603 |  4.433984 |  3.571437 |

Two things to note.

**(a) MetisLB sees the same 13.04 in every arm.** Whatever differs, it does not
differ at the first stats read.

**(b) DiffusionLB's step-1 value is exactly `(period-lag)/period` of sync's:**

    sync   20/20 * 14.774 = 14.774   measured 14.774451
    lag 4  16/20 * 14.774 = 11.819   measured 11.823222
    lag 16  4/20 * 14.774 =  2.955   measured  2.946834

So at the FIRST balancing step the measurement window is exactly `period - lag`
steps, to three digits. That part is understood and expected.

**(c) What is NOT understood: steps 2-4 at lag 16.** If the window were a fixed
4 steps, T_g would stay near 2.95. Instead it goes 2.95 -> 7.08 -> 4.43 -> 3.57,
i.e. implied windows of 4.0, 9.6, 6.0, 4.8 steps. At lag 4 and in sync the value
is stable across steps. The instability appears only when `period - lag` is small.

THIS IS THE OPEN QUESTION. Four independent changes to how loads are gated and
attributed (section 3, items 1 and 6-8) left this swing untouched, which argues
the instability is NOT in what gets billed to which object. Reproduced in every
lag-16 async run measured: 2.95/7.08/4.43/3.57, 2.95/7.73/4.74/3.60,
2.94/7.61/4.80/3.50. Note the shape repeats -- low, high, middle, low -- so it
is not random noise.

Per-step host loads for the same runs (`[PELOAD]`, 8 PEs of process 0):

    sync   LB2: 0.3606 0.5273 0.5776 0.5167 0.6346 0.6398 0.6604 0.5969  sum 4.5139  max/avg 1.17
    async  LB2: 0.0289 0.0629 0.0650 0.0642 0.0854 0.0859 0.0874 0.0863  sum 0.5661  max/avg 1.23
    (lag 16)

---

## 3. Hypotheses already tested and REJECTED

Do not re-run these.

1. **Absolute-vs-relative units in the cost model.** `DiffusionMetric.C:351-356`
   computes `score = benefit - cost` where `benefit` comes from window-scaled
   `objLoad` but `migrateCost` is in wall seconds from the calibration table.
   Real mismatch, but not the cause: dividing `migrate_*` by 6 and by 12
   (matching the measured 5.3x load shrinkage) changed neither the churn nor the
   runtime — `[1182 625 2 606]` / 1075.5 vs `[1187 636 598 2]` / 1079.5.
2. **Load-dimension selection changing between arms.** `lbResolveLoadMode`
   (LBLoadDim.h:232) uses `comparable()` and `deviceBinds()`, both pure ratios of
   measured quantities, hence scale-invariant. `unexplained()`/`hostShare()` are
   print-only (LBLoadDim.h:267-274) and gate nothing. Every run resolves
   `device by flag` anyway.
3. **The measured steps being unrepresentative ("post-migration transient").**
   Per-step times in the lag-16 async run: steps 36-40 average 1294 ms vs
   1392 ms for steps 21-35 — the measured steps are if anything faster.
4. **Per-PE measurement intervals differing in length.** Asserted and rejected by
   the code owner; DiffusionLB has a job-wide sync, so PEs clear together.
5. **Per-object park-order effects** (objects parking at `AtSyncWait` at different
   times and so accumulating different amounts). Asserted and rejected by the
   code owner.
6. **Per-PE instrumentation toggled by a per-chare event.** REAL DEFECT, FIXED,
   DID NOT FIX THE SYMPTOM. `LBTurnInstrumentOn/Off` flip a PE-wide `statsAreOn`
   but were called per chare from `AtSyncStart` / resume (cklocation.C 2549,
   2733, 2791, 2829). With ~240 elements per PE the first element to reach
   AtSync stopped billing all the others. Moving those toggles to PE-wide points
   (balancer load-read and `ClearLoads`) changed nothing measurable:
   sync `[1180 0 0 0]` 1014.3 vs baseline `[1181 0 0 0]` 1016.5;
   async `[1191 628 2 19]` with T_g 2.95/7.73/4.74/3.60.
7. **Per-object billing gated by the PE-wide flag.** REAL DEFECT, FIXED, HELPED
   ~3%, CHURN UNAFFECTED. `LBDatabase::ObjectStart/ObjectStop` wrapped the
   per-object timers in `if (StatsOn())`, so a correctly attributed sample was
   discarded because a *different* chare on the PE had joined. Billing is now
   gated only by that object's own `LBObj::joinedStep`. Result: sync unchanged
   (`[1177 0 0 0]` 1014.9), async 1060.8 -> **1027.3**, LB4 moves went to 0, but
   LB2 still moves 627 and T_g still swings 2.94/7.61/4.80/3.50.
8. **A build with instrumentation accidentally disabled** produced a STABLE
   T_g (2.91/3.25/3.37/3.45) and near-zero migrations. That is not a fix -- with
   `statsAreOn` never enabled, T_g came from CUPTI alone. Recorded because the
   stability is misleading if someone reproduces it.

---

## 4. Relevant code

- `examples/.../leanmd/leanmd.ci:138` — `lbBegin()` (= `AtSyncStart`) fires when
  `stepCount >= firstLdbStep && (stepCount-firstLdbStep) % ldbPeriod == 0`, so at
  steps 20/40/60/80.
- `examples/.../leanmd/Compute.cc:139-142` — `lbWaitDue()` returns true when
  `stepCount - lbStartStep >= lbLag`, i.e. `AtSyncWait` at step 20+lag. Migration
  happens at that park. With period 20 / lag 16 there are only **4 steps of slack**
  between the park and the next `AtSyncStart`.
- `examples/.../leanmd/Main.cc:126-131` — clamp is `lag <= period - 1`, which
  permits a 1-step measurement window.
- `src/ck-ldb/DiffusionLB.C:714` — the within-node floor:
  `if (avgPE <= 0.0 || maxPE <= avgPE * (1.0 + effMinImbalance)) { no moves }`.
  `effMinImbalance` default 0.10 (`LBManager.h:117`). Both arms clear this.
- `src/ck-ldb/DiffusionPseudo.C:108` — `PseudoLoadBalancing`, the flow rounds;
  the `flowAdjacent` room rule is at lines ~125-141.
- `src/ck-ldb/DiffusionMetric.C:340-356` — object selection, `score = benefit - cost`.
- `src/ck-ldb/LBMachineUtil.C:120-138` / `LBDatabase.C:253-284` — `total_walltime`
  accumulation and `ClearLoads`, which resets object `wallTime`/`gpuTime` and
  `machineUtil`.
- `src/ck-ldb/DiffusionLB.C:1252` — `planReport` prints predicted-vs-current
  imbalance, but ONLY when `remapAbove > 0` (`+LBDiffusionRemapAbove`), which is 0
  by default. No run so far has this data.

## 5. Diagnostics that exist but were never switched on

- `CHARM_LB_RAWLOAD=1` — `DiffusionHelper.C:207`. Dumps raw per-object host and
  device loads before any floor, plus per-PE object counts.
- `+LBDebug 2` — adds `[WITHIN node N] max/avg X under floor Y: no moves`
  (DiffusionLB.C:716) and `[FLOWGATE node N] nbor ...` (DiffusionPseudo.C:146),
  which say exactly which gate refused or allowed each destination.
- `+LBDiffusionRemapAbove <x>` — turns on the plan report in section 4.

Running a sync/async(lag 16) pair with `CHARM_LB_RAWLOAD=1 +LBDebug 2` is the
obvious next step and has not been done.

## 5b. Changes made while investigating (all UNCOMMITTED)

- `src/ck-ldb/LBDatabase.h` — `ObjectStart`/`ObjectStop` no longer gate per-object
  billing on `statsAreOn`; `joinedStep` is the only gate. The PE background
  accumulators (`MeasuredObjTime`/`MeasuredObjGPUTime`) still sit behind it
  deliberately, being PE-level quantities. **This is the one change worth
  keeping** — it is a correctness fix and buys ~3% on async.
- `src/ck-core/cklocation.C` — the four per-chare `LBTurnInstrumentOn/Off` removed.
- `src/ck-ldb/DistBaseLB.C`, `src/ck-ldb/CentralLB.C` — a PE-wide
  `LBTurnInstrumentOn()` added after `ClearLoads`, pairing with the existing
  PE-wide OFF where the strategy reads the loads. Measured as a no-op; keep or
  revert as preferred.

## 6. Where the logs are

Under `examples/charm++/cuda/gpudirect/leanmd/ranklogs/`:

- `tsdbg_22106974_fix3_sync/`, `tsdbg_22106974_defe_async/` — sync and async lag 16,
  uncalibrated (no `+LBCostConfig`), full `rank_0..3.log`.
- `weak_N1_sync/`, `weak_N1_async/` — sync and async **lag 4**, with the cost table.
- `scale-22109962.out`, `scale-22109968.out` in `examples/charm++/cuda/gpudirect/` —
  N=1 and N=2 weak sweep summaries.

Each `rank_0.log` has `Step N Benchmark Time`, `[PELOAD]` per-PE loads, the
`load dimension:` lines, and `cross node migrations` per LB step.

## 7. Other results from the same session (context, not the question)

- leanmd weak N=1, with cost table: noLB 1438.4, sync 1107.3, async(lag16) 1185.2.
- leanmd weak N=2: noLB 1649.3, sync 1570.4, async 1592.6. The balancer reports
  `host 3%, device 48%, launch 11%` of the interval — ~40% of a 2-node step is
  explained by no measured dimension, so no placement can win much there.
- sph2d weak: N=1 noLB 21.893 / sync 14.725 / async 14.926;
  N=2 noLB 23.317 / sync 15.435 / async 16.486.
- Cost model: `inter_node` tier was measured for the first time on 2 nodes
  (job 22109113): alpha 2.559e-05 -> 5.195e-05, beta 3.992e-11 -> 5.421e-11,
  R2 0.9885. Previously copied from `ipc_cross_gpu`. Table at
  `examples/charm++/cuda/gpudirect/sph2d/lbcost.delta-a40.2node.conf`.
- Two bugs found on 2 nodes: a migration batch of 32 segfaults at the 1 MB device
  point (batch 8 is fine, batch 32 is fine at 1 node); and cross-node migration of
  a host-only chare corrupts the heap on the destination — glibc `malloc_printerr`
  -> `_int_free` inside LCI `free_ctx_and_signal_comp` on the send-completion path.
