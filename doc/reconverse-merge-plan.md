<!-- Living document: updated as stages complete. Lives on the reviewed
     reconverse line so plan changes go through the same lightweight
     review as code. Item 21 retires this file's subject (and most of
     the file) at the rename milestone. -->

# Reconverse merge — staged plan and deferred items

Maintained by Kale + Claude sessions. Created 2026-08-26, after PR #3944
went fully green (Actions + CircleCI). Items are anchored to trigger
events, not dates: each stage unlocks when its predecessor merges.

## Now (unlocked)

| # | Item | Trigger / owner |
|---|---|---|
| 1 | ~~Group reviews + merges **#3944**~~ | DONE 08-27: squash-merged `bc819bcf2` |
| 2 | ~~Squash at merge~~ | DONE 08-27 (squash-merge) |

## After #3944 merges

| # | Item | Notes |
|---|---|---|
| 3 | ~~Retarget/merge **#3945**~~ | DONE 08-27: squash-merged `24865a675`; post-merge push run green in 4m48s |
| 4 | ~~Required status checks on the reviewed line~~ | DONE 08-27: all four reconverse jobs required, strict mode on |
| 5 | Reduction fix (#3939) to the reviewed line, heavy stress enabled in the same PR | **PR #3947** (green, in review); #3942 remains the classic-main vehicle for item 8 |
| 6 | ~~Record-replay repairs to the reviewed line~~ | DONE by subsumption: already present via #3944's ck.C graft (verified 08-27); #3943 remains the classic-main vehicle for item 8. Verification found the token-pool comment-out — proper guard in **PR #3948** |
| 7 | Re-pin `AUTOFETCH_RECONVERSE_TAG` to reconverse **main** | THIS PR: pin -> `f3f4110` (#207 squash had missed the CMK_RECONVERSE define; reconverse#210 carried it to main 08-27) |
| 8 | Merge #3942/#3943 to classic **main** too, then cut the freeze tag (last classic release) | order matters: fixes in before tag |
| 8b | **Classic main is maintenance-mode after the freeze tag, not dead** | Policy (Kale, 08-27): bugfixes keep landing on main for users not yet on reconverse (known dependent: Quinoa/CFD, Aditya Pandare — fix current bug jointly, then migrate Quinoa to reconverse as an external pilot). Main keeps full CI incl. the #3946 ARM jobs; feature work on the reviewed line only |
| 8a | Silence CircleCI's "no configuration found" status on the reviewed line | DONE 2026-08-27: Eric turned CircleCI off project-wide; the interim stub is removed. Classic main's ARM coverage moved to Actions (PR #3946: netlrts-linux-arm8, smp variant, mpi-linux-arm8 on ubuntu-24.04-arm) |

## After the core is stable on the reviewed line (deferred feature series, in order)

| # | Item | Notes |
|---|---|---|
| 9 | **GPU device-to-device (LCI) + Kokkos** PR series | restores the branch's HAPI unification; revisit the `cudahybridapi`→`hybridapi` rename here, where the HIP path actually lands |
| 10 | **Shrink/expand** | |
| 11 | **GPU-aware load balancing** | |
| 12 | **LCW comm backend** (MPI-backed alternate to LCI — comparison + fallback where LCI is unavailable/inefficient) | deliberately post-core (Kale, 2026-08-26); additive behind the comm_backend interface |
| 13b | **Object queues (ckobjQ) on reconverse** | with/after the locmgr changes (item 15 is the natural vehicle); classic-only today, token-pool registration guarded via PR #3948 |
| 13 | **Windows multicore** via a null/self comm backend in reconverse (no LCI/libfabric dep) | raise with Aditya; NAMD-desktop use case; frozen classic serves it meanwhile |

## Parallel / independent

| # | Item | Notes |
|---|---|---|
| 13a | **Grow the CI tier toward real coverage** | standing item, good student-sized increments: chkpt, sdag, zerocopy, io, dynamic insertion/deletion, partitions ... each addition = one reviewed commit adding a dir to TEST_DIRS (and fixing what it flushes out). The current 5-test tier is a floor, not the support surface. |
| 14 | **Parity matrix document** (classic vs reconverse: works / planned / tombstoned) | LCW + Windows-multicore are "planned", message-logging FT and dead code are tombstones |
| 15 | Joint **locmgr review** of Aditya's e417584dd (notes: recharm/NOTES-locmgr-review.md) | Kale + Claude |
| 16 | Reconcile **#3939** with Aditya | Kale points him at the issue |
| 17 | Investigate sanctioned **HPC-site runners** (OLCF/NCSA) for CI; until then reviewers run on machines of choice (PR template field) | |
| 18 | 2-node InfiniBand test tier on Anvil (inside allocation, `-N 2`, `--mpi=pmi2`) | network tests need multinode jobs, not login nodes |
| 19 | Record-replay follow-ups: CkExit flush hook; threaded-entry replay validation | |
| 20 | Rename `reviewed-with-reconverse` → `main` | at NAMD + ChaNGa parity |

## Closure guarantee (mechanical)

| # | Item | Notes |
|---|---|---|
| 22 | **Migration ledger**: `doc/reconverse-migration-ledger.tsv` + `doc/check-migration-ledger.sh` | Every file differing between the reviewed line and `reconverse-specific-build` carries a disposition (plan item, PR, superseded, tombstone, needs-judgment). The script fails on any undispositioned file — including new commits landing on the old branch — so nothing migrates silently or gets forgotten. Migration is provably complete when the check passes with only tombstone/superseded rows remaining. Resolve the needs-judgment rows (locmgr/cklocation.C via the queued joint review; conv-ccs; pup) as their reviews happen. |

## At the rename milestone (same trigger as #20)

| # | Item | Notes |
|---|---|---|
| 21 | **The great deletion**: remove classic from the renamed main | The two-way guards are transitional scaffolding, kept so the reviewed line has a working classic fallback during app migration (and so old CI machine-checks that reconverse additions change nothing for classic). Removal is mechanical by design: `unifdef -DCMK_RECONVERSE=1` collapses every `#if CMK_RECONVERSE` source guard; cmake `if(NOT RECONVERSE)` branches delete wholesale with the classic-only dirs (QuickThreads, charmrun, classic conv-core, netlrts/verbs/ucx/mpi layers); two-way boundary headers (ckrdmadevice.h etc.) revert to single-body. Windows-multicore users stay on the frozen classic release until item 13's null backend lands — that's independent of this deletion. |
