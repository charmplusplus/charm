# Async LB measurement state across migration

## Defect

`lbJoinStep()` closes the object's host measurement gate (`LBObj::joinedStep`)
and adds its identity to the PE-local CUPTI exclusion set. Both gates are meant
to stay closed until the object's wait releases it.

The runtime permits migration before `AtSyncWait()`. A destination creates a
fresh `LBObj` with an open gate, and a first-time destination has no CUPTI
exclusion for the arriving object. Previously, neither gate was restored.
Objects that migrated could therefore accumulate load during the async lag,
while unmoved objects remained excluded.

The handoff's statement that migration happens at the park is incorrect for
the default runtime. In `tsdbg_22106974_defe_async/rank_0.log`, the first
DiffusionLB round finishes between simulation steps 41 and 42, before the
lag-16 park at step 56. The park-only behavior is an optional diagnostic mode
(`CHARM_LB_MIGRATE_AT_PARK_ONLY`).

## Fix

`CkMigratable::lbMeasurementClosed` preserves the measurement state independently
of `lbStepPending`: the latter can clear before the application reaches its wait.
A shared setter updates the object state, the destination LB database gate, and
the destination PE's CUPTI exclusion set.

The state travels through migration PUP and is restored in the same-process
migration path, which transfers the live object without PUP. Restoring an open
state also removes a stale CUPTI exclusion when an object returns to a previous
PE. All wait-release paths use the same setter, including the arrival-epoch
fallback.

## Validation

The runtime, load-balancer modules, LeanMD, and the `async_arrival` regression
executable were rebuilt. The regression checks that migration preserves the
closed gate, that completing an async step does not reopen it before the wait,
and that resume reopens it.

Delta allocation 22114870 used one A40 node (`gpub097`), four GPUs, and
four processes with eight PEs each. All three full LeanMD runs completed
100 steps with exit code 0 and identical final energy, 3.3802584099E-13.

| Run | Device bound T_g at DiffusionLB rounds 1–4 (s) | Reported cross-node migrations | Mean steps 42–100 (ms) |
| --- | --- | --- | --- |
| Before, async lag 16 | 2.948561, 6.820992, 4.229378, 4.411368 | 1362, 612, 599, 60 | 1137.1 |
| Fixed, async lag 16 | 2.910671, 3.110019, 3.161735, 3.256728 | 1363, 0, 0, 0 | 1117.6 |
| Fixed, sync | 14.778387, 15.666395, 15.837938, 15.966461 | 1368, 0, 0, 0 | 1106.3 |

The fixed async bounds are approximately 4/20 of sync's, consistent with the
four-step measurement window. The swing and subsequent migration churn are
gone. These are single runs; the small timing differences do not establish a
speedup. Raw process-0 GPU load at the second DiffusionLB round fell from
11.223490 s to 2.985739 s. No process reported zero-load objects in these runs.

A 4x4x4, 40-step GPU smoke test also passed with migrations 232, 20, 1, 1.

The final regression harness passed four rounds with both same-process and
cross-process migration, including a split wait taken after LB completion.
Follow-up allocation 22115207 used two GPUs on one A40 node (`gpub079`), with
one GPU per process in the cross-process checks. The pre-fix control links the
saved pre-fix `cklocation.C` implementation into the regression executable.
It fails with exit code 134 and the precise assertion
`[TEST] migration reopened the measurement window`; the fixed executable passes.
The original benchmark allocation ended, and the follow-up allocation was
released after validation.

During test development, an override of `ckJustMigrated()` omitted its base
call, suppressing array-arrival listener notifications and causing reduction
hangs. The final test calls `CBase_Blk::ckJustMigrated()` and passes. The original
unmodified test also passed with both the old and fixed runtime implementations.

The runner scripts and complete logs are under
`examples/charm++/cuda/gpudirect/leanmd/measurement-test/`.

The LeanMD comparison uses 8x8x8 cells, 100 steps, LB period 20, async lag 16,
MetisLB followed by DiffusionLB, four processes with eight PEs each, explicit
NUMA pinning, and a 2048 MB GPU pool. Both comparison executables use the same
environment and flags. The pre-fix executable was preserved before rebuilding.

Build logs are under `/tmp/async-lb-measurement/`.
