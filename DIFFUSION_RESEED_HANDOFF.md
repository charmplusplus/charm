> **Status 2026-09-17.** The policy below was measured on the 2-node weak grid
> (job 22133779) at 1873 ms/step against 1576 unbalanced: the mandatory
> destination attachment (item 4) starved the donors once their face computes
> were gone. It was replaced by an unconfined, score-ranked restart on the same
> trigger (commit "DiffusionLB: reseed by score at a structural end"); items 4
> and 5 no longer apply. Measured at 1442 ms/step (job 22135839).

# Diffusion locality-preserving reseeding: implementation and testing handoff

## Status

Implemented in `src/ck-ldb/DiffusionMetric.C` and `.h`. The production
`moduleDiffusionLB` target builds, and 14 standalone decision-code scenarios
pass. **The new policy has not been run on GPUs or validated on the full
stencil benchmark.** Do not transfer the broad grow-any performance result to
this narrower policy without testing it. Changes are uncommitted; preserve the
other existing work in this checkout.

## Problem and evidence

LeanMD's movable computes communicate through fixed cells. After migration,
both partner cells can be remote, leaving a compute with no local graph edges.
The old selector accepts one compute, then requires subsequent moves to grow
through its local communication neighbors. Its empty frontier stops selection
even with affordable work and receiver quota remaining.

Two-node A40 matched comparison, job 22132598:

| metric | old default | broad grow-any override |
| --- | ---: | ---: |
| mean steps 42–100 | 1530.5 ms | 1391.8 ms |
| total application | 148.72 s | 141.02 s |
| final measured GPU max/average | 1.395 | 1.242 |
| migrations per round | 1471, 508, 865, 3 | 915, 898, 564, 287 |

Default's final donors accepted one move apiece, left 4.4–5.3 seconds of desired
load unshed, and reported no cost, receiver, or memory rejections. See
`N2_GROW_ANY_RESULTS.md` and `leanmd/grow-test/summary.json` for full evidence.

## Implemented policy

1. Normal seed selection and connected frontier growth retain their scoring.
2. A reseed is possible only if the available frontier has **no positive-load
   movable object**. This classification happens before quota, allowed-mask,
   capacity, and cost filtering: rejection of a live frontier does not justify
   jumping elsewhere.
3. An empty frontier has an additional guard: none of the objects accepted for
   this destination during the current round may have had a positive-load,
   movable local neighbor. This distinguishes independent computes from an
   exhausted connected stencil component. Empty-frontier reseeding additionally
   requires a calibrated cost model. A frontier containing only fixed/zero-load
   objects can still reseed, including on the legacy uncalibrated path.
4. Every reseed **must already communicate with the destination**. Unlike the
   original initial-seed rule, this requirement never falls back to arbitrary
   objects when no attached candidate exists.
5. Among admissible reseeds, prefer an object sharing an affinity endpoint with
   work already accepted for that destination; within the same preference class,
   use the existing score. Affinity endpoints are local fixed objects and remote
   object identities. Full `LDObjKey` identity includes the object manager.
6. Calibrated reseeds require positive net benefit and retain quota, allowed-mask,
   receiver load, memory, and migration/communication cost checks. Existing
   `updateState` accounting still updates costs after every accepted move.

Remote endpoint mobility is **not available in local stats**. Remote endpoints
are affinity hints, not assertions that the remote object is fixed. Sharing an
endpoint is a preference, not a new estimated cost. If no affordable shared-
endpoint candidate exists, another affordable destination-attached seed is
allowed. These details are intentional and should be evaluated in testing.

The rule is enabled by default; `CHARM_DIFFUSION_GROW_ANY=1` remains the explicit
broad ablation. **Unset that variable when testing the new rule.** No new global
barrier, communication exchange, or calibration constant was introduced.

## Diagnostics

With `+LBDebug 2`, reseed attempts print:

```
[RESEED node N] nbor I reason=independent-empty-frontier -> obj O score S
[RESEED node N] nbor I reason=fixed-or-zero-frontier -> obj O score S
```

`nbor` is a neighbor-list index, not its process ID; use the `toSendLoad` lines
to map it. `obj=-1` means no admissible attached reseed was found. Costs rejected
during the reseed scan are filtered before selection, so do not use the old
`rejectedCount` alone to classify a failed reseed. Nonnegative `obj` is a candidate
accepted by the subsequent common acceptance path. Process/GPU IDs called
"nodes" by Diffusion are not physical-node IDs.

## Local validation

```
make -C tests/charm++/load_balancing/lbdriver check-reseed
cmake --build multicore-linux-x86_64-cuda --target moduleDiffusionLB -j 4
make -C examples/charm++/cuda/gpudirect/leanmd -W Main.o
git diff --check
```

The new `reseed_test.C` links the real selection metric with existing
`lbsim_stub.C` runtime stubs; it is safe to run on the login node. Cases cover
remote-partner reseeding, shared-endpoint preference, no destination attachment,
negative benefit, fixed local frontiers, exhausted connected components,
quota/host-capacity/memory/allowed-mask/cost barriers, and the broad override.
The compile emits preexisting macro-redefinition warnings. The module also
emits the preexisting missing-return warning in `DiffusionJSON.h`.

## GPU comparison to run on Delta

Use only A40, two physical nodes, four processes/GPUs per node, eight PEs per
process, 16 allocated CPU cores per process, and the existing reversed-NUMA
`rankwrap.sh`. Use `+gpupool` with 2048 MB arenas and preserve `LD_LIBRARY_PATH`.
Set `FI_MR_CACHE_MONITOR=disabled`, `PMI_MAX_KVS_ENTRIES=8192`, and `ulimit -c 0`.
No application binaries should run on the login node.

Compare on the same allocation, sequentially:

1. **Old default:** saved `leanmd/join-test/leanmd-trace`, grow-any unset.
2. **New targeted policy:** newly linked `leanmd/leanmd`, grow-any unset.
3. **Broad override:** the same newly linked binary with grow-any set to 1.

All arms: weak grid `16 8 8`, 100 steps, first LB 20, period 20, local compute
mapping, gradient density, Metis then Diffusion, communication enabled, GPU
dimension forced, same `sph2d/lbcost.delta-a40.2node.conf`. Use
`CHARM_GPU_LOAD_AUDIT=1`, `LBDEBUG=2`, and keep `LEANMD_JOIN_TRACE=1` consistent
with the saved old binary's preceding experiments. Take additional matched
repetitions or trace-disabled comparisons if the performance difference is small.

**Avoid the stale-binary trap:** `join-test/scale.sh` and `grow-test/scale.sh`
explicitly launch the saved OLD `join-test/leanmd-trace`. Relinking the normal
application does not update that snapshot. Copy the harness into a fresh
directory and explicitly select the intended executable for each arm. Give each
arm its own rank-log directory; `scale.sh` deletes its selected directory.
Do not overwrite old evidence. Existing two-arm `grow-test/run.sbatch` takes
about five minutes but does not run this new implementation.

Inspect:

- Successful energy checks and completion of all 100 steps.
- `[RESEED]` attempts and remaining unshed load on overloaded donors.
- Per-GPU attributed work and maximum/average **in the next interval**.
  `T_g` is not the measured maximum. Step-100 moves have no following interval.
- Total application time and step-window means; do not infer a speedup just
  from more moves or shorter force joins.
- Communication cut, cross-node traffic, pool growth, and migration volume.

Desired outcome: targeted reseeding avoids one-move stalls, improves balance
and runtime toward the broad override, while avoiding its unconstrained
placement changes. This outcome is not guaranteed by the unit tests.

## Stencil locality regression tests

Build the standalone simulator (`make lbsim-standalone` in `lbdriver`) for old
and new decision code in separate temporary directories/checkouts; do not reset
the dirty working tree. Save each run's `lbsim.json` before the next run.
The old metric sources can be retrieved read-only with `git show HEAD:...`;
verify the baseline revision used. Match all other source and configuration.

Start with the existing 32×32, eight-virtual-node, six-neighbor, hot-disc cases:

```
./lbsim-standalone 32 32 8 4 4096 20 metis 12 +LBDiffusionCommOn +LBDiffusionNumNbors 6
```

Also test calibrated costs using `+LBCostConfig` with the existing synthetic
cost file, and the simulator's host/device cross-load cases documented in
`lbsim.C`. Compare connected components per partition, edge cut, remaining
imbalance, and migrations against the old default, not only grow-any. Use
`analyze.py`/`plot_map.py` to inspect saved maps. An exhausted connected component
must not begin a new detached piece because of this change.

## Scope and remaining risks

This fixes a candidate-selection restriction, not the entire balancing model.
It does not prove optimal placement, an accurate cost model, or a complete
explanation of LeanMD's two-node scaling. Destination attachment and shared
endpoints preserve a useful locality constraint but cannot guarantee a globally
connected partition in all graphs. The conservative independent-piece guard
may still stop some beneficial moves; inspect its effect before relaxing it.

Files for this implementation: `DiffusionMetric.C`, `DiffusionMetric.h`,
`tests/charm++/load_balancing/lbdriver/reseed_test.C`, that directory's `Makefile`,
and this handoff. Earlier LeanMD tracing and runtime changes predate this task.
