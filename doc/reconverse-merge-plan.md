<!-- Living document: updated as stages complete. Lives on the reviewed
     reconverse line so plan changes go through the same lightweight
     review as code. Item 21 retires this file's subject (and most of
     the file) at the rename milestone. -->

# Reconverse merge — staged plan and deferred items

Maintained by Kale + Claude sessions. Created 2026-08-26, after PR #3944
went fully green (Actions + CircleCI). Items are anchored to trigger
events, not dates: each stage unlocks when its predecessor merges.

Status 2026-09-02: the core series and the whole GPU series (9.1-9.3) are
merged, the freeze tag `v8.0.2` is cut, and **the next action is the
item-21 PR** (endgame step 3). The renames behind it wait on Kale's
announcement email. A GPU follow-on section below tracks work opened by
running the merged support rather than by merging it.

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
| 5 | ~~Reduction fix (#3939) to the reviewed line, heavy stress enabled~~ | DONE 08-28: **#3947** merged `601472d57` |
| 6 | ~~Record-replay repairs to the reviewed line~~ | DONE by subsumption: already present via #3944's ck.C graft (verified 08-27); #3943 remains the classic-main vehicle for item 8. Verification found the token-pool comment-out — proper guard in **PR #3948** |
| 7 | ~~Re-pin `AUTOFETCH_RECONVERSE_TAG` to reconverse **main**~~ | DONE, and re-done as reconverse moved: `f3f4110` (#3949) -> `0673ee8` (#3956, picks up reconverse#211 `CmiNodeRankOnPhysicalNode`) -> `2c50813e7` (#3959, picks up reconverse#212, the `CmiCheckAffinity` startup hang). Standing obligation: every reconverse merge we depend on needs a pin bump here, and a stale build directory keeps the old pin (see the cache-seed guard in `cmake/fetch_reconverse/CMakeLists.txt`) |
| 8 | ~~Merge #3942/#3943 to classic **main**, then cut the freeze tag~~ | DONE 08-30: tag **`v8.0.2`** at `935d1441d` is the final classic release. Announcement email to users is Kale's, and gates the renames in step 4 below |
| 8b | **Classic main is maintenance-mode after the freeze tag, not dead** | Policy (Kale, 08-27): bugfixes keep landing on main for users not yet on reconverse (known dependent: Quinoa/CFD, Aditya Pandare — fix current bug jointly, then migrate Quinoa to reconverse as an external pilot). Main keeps full CI incl. the #3946 ARM jobs; feature work on the reviewed line only |
| 8a | Silence CircleCI's "no configuration found" status on the reviewed line | DONE 2026-08-27: Eric turned CircleCI off project-wide; the interim stub is removed. Classic main's ARM coverage moved to Actions (PR #3946: netlrts-linux-arm8, smp variant, mpi-linux-arm8 on ubuntu-24.04-arm) |

## After the core is stable on the reviewed line (deferred feature series, in order)

**Validation protocol for the GPU-touching series (items 9, 10, 11), per
Kale 2026-08-28:** CI provides build gates and GPU-free tests only (the
path-filtered `reconverse-cuda` workflow builds both CUDA paths with
install/naming assertions; no hosted runner has a GPU). Runtime evidence
comes from people: (a) the author side ships every stage with at least
one recorded real-GPU run (the Anvil pre-validation loop; e.g. 9.1's
A100 smoke), and (b) reviewers test on the GPU types and networks of
their choice — AMD/Frontier being the one that actually exercises the
portability claim — recording where they ran in the PR template's
HPC-site field, and **not approving until satisfied**. That approval
discipline is deliberately manual; the template field makes it visible
rather than enforced. Item 10 additionally needs multi-node runs with
job resizing — login-node or single-node evidence is not sufficient
there.

| # | Item | Notes |
|---|---|---|
| 9 | ~~**GPU device-to-device (LCI)** PR series~~ | DONE: 9.1 HAPI portable core (**#3951**, `c10ffa01a`), 9.2 device-to-device (**#3955**, `850ede957`), 9.3 examples + CI (**#3958**, `d535b1d33`). Shepherded by Ritvik. Kokkos and the `cudahybridapi`->`hybridapi` rename were NOT part of the series and remain open |
| 10 | **Shrink/expand** | |
| 11 | **GPU-aware load balancing** | Not started on this line. `src/ck-ldb/` has no GPU strategy; the hooks are merged ahead of it (`HAPI_CUPTI_LB` behind an ifdef nothing defines, the comm buffer's LB region, `+gpulbbuffer`). This is the one capability Kale calls out as missing when telling application developers the GPU support is usable |
| 12 | **LCW comm backend** (MPI-backed alternate to LCI — comparison + fallback where LCI is unavailable/inefficient) | deliberately post-core (Kale, 2026-08-26); additive behind the comm_backend interface |
| 13b | **Object queues (ckobjQ) on reconverse** | with/after the locmgr changes (item 15 is the natural vehicle); classic-only today, token-pool registration guarded via PR #3948 |
| 13 | **Windows multicore** via a null/self comm backend in reconverse (no LCI/libfabric dep) | raise with Aditya; NAMD-desktop use case; frozen classic serves it meanwhile |

## GPU follow-on (opened by the overdecomposition campaign, 2026-09-01/02)

Work that came out of running the merged GPU support rather than merging
it. Frontier (MI250X/Slingshot) drives the campaign; Anvil (A100/IB)
provides the NVIDIA half, which Frontier structurally cannot.

| # | Item | Notes |
|---|---|---|
| 9a | **Device registration leak** (issue **#3960**) | Device zerocopy registered the buffer on both ends of every message and never deregistered; long GPU-direct runs die with ENOSPC from `fi_mr_regattr` (~1000 iterations on Slingshot). **PR #3961** carries fix (a) release-on-completion with piggybacked acks, fix (b) `CkDeviceBufferRegister` register-once, the registered device pool (#3962, merged into 3961's branch), per-device arenas, and the manual text. Open, review required |
| 9b | **`CkDeviceMalloc` / `CkDeviceFree` API** | New public surface inside #3961: arenas over the existing buddy allocator, registered lazily and whole on first network use. Kale + Aditya to review the API deliberately rather than let it merge on the correctness merits |
| 9c | **Examples repack send buffers unsafely** (issue **#3963**) | No shipped gpudirect example passes a source callback, and `jacobi2d` repacks its send buffers every iteration with nothing guaranteeing the neighbour's get has finished. Fix is double-buffering or the callback; the manual text in #3961 documents the rule |
| 9d | **`hapiGetStream` deprecation** | Decided in the 2026-09-01 group meeting: users create their own streams; a return queue may come later. Small deprecation surface — one implementation, one AMPI export, one test, one manual entry. Not yet done |
| 9e | **Stream-ordered delivery as an entry attribute** | Measured -33%, but the prototype drops the QD bracket (quiescence reachable with copies in flight) and runs the entry method nested via `enqueueNcpyMessage` -> `CmiHandleMessage`. Needs `CsdEnqueue` delivery plus a per-stream QD sentinel, and charmxi work. General-purpose, not stencil-specific |
| 9f | **Host path: bundle dereg acks** (issue **#3964**) | The device path's piggybacked acks are ahead of the host path, which still sends a whole `NcpyOperationInfo` per buffer |
| 9g | **User documentation** | Kale's standing rule: user-facing docs change with the change. #3961 does this for registration and buffer reuse. Still owed: the `hapiGetStream` deprecation, the hardware limits (queues 32 NVIDIA / ~8 AMD, concurrent kernels, copy engines), and `+gpushm` |

Campaign results that bear on the plan rather than the code: same-node
IPC (`+gpushm`) measures **slower** than the RDMA-get path at 2-16 KB, so
the transfer-mode design is not settled; overdecomposition pays at
**2 chares per device** and costs nothing up to 32; and the concurrency
ceiling for issue-bound chares is the single PE thread, not the hardware
queues (the knee is identical at 4 and 8 queues) — remedied by several
PEs per GPU. NVIDIA at 32 queues is unmeasured and is Anvil's to answer.

## Parallel / independent

| # | Item | Notes |
|---|---|---|
| 13a | **Grow the CI tier toward real coverage** | standing item, good student-sized increments: chkpt, sdag, zerocopy, io, dynamic insertion/deletion, partitions ... each addition = one reviewed commit adding a dir to TEST_DIRS (and fixing what it flushes out). The current 5-test tier is a floor, not the support surface. |
| 14 | **Parity matrix document** (classic vs reconverse: works / planned / tombstoned) | LCW + Windows-multicore are "planned", message-logging FT and dead code are tombstones |
| 15 | Joint **locmgr review** of Aditya's e417584dd (notes: recharm/NOTES-locmgr-review.md) | Kale + Claude |
| 16 | Reconcile **#3939** with Aditya | Kale points him at the issue |
| 17 | Investigate sanctioned **HPC-site runners** (OLCF/NCSA) for CI; until then reviewers run on machines of choice (PR template field) | |
| 18 | 2-node InfiniBand test tier on Anvil (inside allocation, `-N 2`, `--mpi=pmi2`) | network tests need multinode jobs, not login nodes. Prerequisite found and fixed 09-01: multi-node reconverse hung in `CmiCheckAffinity` whenever physical node 0 held exactly one PE (reconverse**#212**, pinned in via **#3959**). Until then no 2-node reconverse run had ever succeeded on Anvil |
| 19 | Record-replay follow-ups: CkExit flush hook; threaded-entry replay validation | |
| 20 | Rename `reviewed-with-reconverse` → `main` | at NAMD + ChaNGa parity |

## Closure guarantee (mechanical)

| # | Item | Notes |
|---|---|---|
| 22 | **Migration ledger**: `doc/reconverse-migration-ledger.tsv` + `doc/check-migration-ledger.sh` | Every file differing between the reviewed line and `reconverse-specific-build` carries a disposition (plan item, PR, superseded, tombstone, needs-judgment). The script fails on any undispositioned file — including new commits landing on the old branch — so nothing migrates silently or gets forgotten. Migration is provably complete when the check passes with only tombstone/superseded rows remaining. Resolve the needs-judgment rows (locmgr/cklocation.C via the queued joint review; conv-ccs; pup) as their reviews happen. |

## Endgame (accelerated per Kale, 2026-08-28)

Item 21 no longer waits for NAMD/ChaNGa parity: it executes after the
freeze tag on main. Order:

1. ~~Merge the in-flight PRs: #3951 (GPU stage 9.1), #3952 (ci.yaml
   action upgrades on main).~~ DONE.
2. ~~Kale tags main — the immutable last classic release.~~ DONE
   2026-08-30: **`v8.0.2`** at `935d1441d`, announced as the final
   release of classic Charm++.
3. **Item-21 PR** on the reviewed line (below). **<- NEXT ACTION.** Must include deleting
   `.github/workflows/ci.yaml`: its trigger is `branches: [main]`, so
   after step 4 it would otherwise start running the classic matrix on
   every PR to the renamed branch.
4. Rename `main` -> `classic` (protection moves with it) and set the
   DEFAULT branch to `reviewed-with-reconverse`. Deliberately do NOT
   rename this line to `main` yet (amendment, Kale 2026-08-28): with no
   branch named `main`, every stale consumer -- old clones pulling,
   scripts pinned to origin/main -- fails loudly ("couldn't find remote
   ref main") instead of silently receiving a different runtime's tree.
   No open PRs may target old main at this point.
5. Announcement to users: `main` no longer exists; classic dependents
   run `git checkout classic`; new development and releases are on the
   default branch `reviewed-with-reconverse`.
5a. Adopting the name `main` for this line happens later as its own
   small, announced event, once the loud-failure window has done its
   job. Until then the reconverse workflows' branch triggers need no
   changes, and the deleted classic ci.yaml could never have fired
   anyway (its `branches: [main]` matches nothing here).
6. Release: a release tag on the new main when the group deems it
   ready. Releasing does not require waiting for anything beyond this
   sequence; NAMD/ChaNGa validation continues in parallel and gates
   nothing here.

## At the rename milestone (same trigger as #20)

| # | Item | Notes |
|---|---|---|
| 21 | **The great deletion**: remove classic from the renamed main | The two-way guards are transitional scaffolding, kept so the reviewed line has a working classic fallback during app migration (and so old CI machine-checks that reconverse additions change nothing for classic). Removal is mechanical by design: `unifdef -DCMK_RECONVERSE=1` collapses every `#if CMK_RECONVERSE` source guard; cmake `if(NOT RECONVERSE)` branches delete wholesale with the classic-only dirs (QuickThreads, charmrun, classic conv-core, netlrts/verbs/ucx/mpi layers); two-way boundary headers (ckrdmadevice.h etc.) revert to single-body. Windows-multicore users stay on the frozen classic release until item 13's null backend lands — that's independent of this deletion. |
