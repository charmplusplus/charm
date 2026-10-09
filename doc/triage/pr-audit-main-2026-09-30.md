# Open PRs against `main` — relevance to `reviewed-with-reconverse` (audit 2026-09-30)

Scope: every OPEN PR in charmplusplus/charm whose base is `main`: **148**, not the ~58 expected. 86 of them are the 2019 Gerrit bulk import (created 2019-05-08, original dates 2013-2018), and 29 are drafts.
Branches compared: `origin/reviewed-with-reconverse` at `ebdc49c66`, `origin/main` at `7cc58941e`.
Read-only audit: nothing was commented on, closed, labelled or checked out.

Method: I read each PR's body and file list. I exported both branch trees with `git archive` and ran `git apply --check`, forward and reverse, with every PR diff against both trees. No PR reverse-applies, so no diff is present byte for byte on either branch. The A rows were found by symbol and commit searches (`git log -S`, `git grep`). "Applies cleanly" in a reason means the forward check passed on the reviewed line. Two diffs (#3392, #2209) are larger than GitHub's 20k-line diff limit and were judged from their file lists. Facts about the reviewed-line build that drove many rows:

- Under `RECONVERSE`, `src/conv-core`, `src/QuickThreads`, boost-context, classic `conv-ccs` and `cmitls.C` are not compiled (cmake/converse.cmake, CMakeLists.txt L998). So any change only to those is B.
- TCharm and AMPI are gated `if(NOT RECONVERSE)` ("not yet ported"). AMPI PRs are marked C where the change is still wanted, but they cannot be exercised on reconverse until TCharm/AMPI are ported.
- Neither branch has any workflow except `ci.yaml` (plus `reconverse-ci.yaml`, `reconverse-cuda.yaml` on the reviewed line). PRs that edit `spack.yml`, `mpi_linux_smp.yml`, `ucx.yml` and similar edit files that no longer exist.
- `src/util` (pup etc.), `TopoManager`, `ck-core`, `ck-perf`, `ck-ldb`, charmxi, hwloc and ck-libs (except TCharm/AMPI) build on both runtimes.

Author status: the current contributors are ritvikrao, adityapb, ericjbohm and lvkale. Everyone else is marked former, including the original authors of the Gerrit imports (named in parentheses) and the LGTM bot.

## Counts

| bucket | count | of which drafts | by former authors |
|---|---|---|---|
| A: already on the reviewed line (equivalent) | 10 | 2 | 9 |
| B: obsolete (classic-only machinery, or code gone) | 32 | 9 | 29 |
| C: relevant, still wanted | 57 (22 trivial, 22 moderate, 13 hard) | 11 | 52 |
| D: unclear, needs the author or a decision | 49 | 7 | 48 |
| **total** | **148** | **29** | **138** |

## PRs by lvkale (flagged separately)

- **#4013** CkTreeCacheManager (C, trivial): opened 2026-09-28 against `main`, which is about to be frozen. It applies cleanly to the reviewed line; retarget it with `gh pr edit 4013 --base reviewed-with-reconverse`.
- **#3937** trace-summary message bytes (C, trivial): the same situation, applies cleanly; retarget.
- **#3676** nodegroup-message priority in the classic scheduler (B): the code is classic `convcore.C`. The problem it fixes (#3674) probably also exists on reconverse. Its registered scheduler (`contrib/reconverse/src/scheduler_registered.cpp`) polls `pollConverseNodeQueue` and `pollNodePrioQueue` at weight 1, against weight 16 for the PE-local queues. The fix therefore needs a new reconverse PR; retargeting this one will not help.

## Other current-contributor PRs

- ritvikrao: #3910 (A, superseded by #3944) and #3904 (C, moderate).
- ericjbohm: #3869 (C), #3859 (B), #3858 (D), #3768 (B) and #3589 (C).

Everything else (138 PRs) is by former or external contributors.

## Closing the rename gap

Step 4 of the endgame plan says "No open PRs may target old main at this point". That means the 148 PRs must be retargeted or closed first, whatever their bucket. When the base branch is renamed, GitHub moves open PRs to `classic` automatically. After that, only a manual retarget moves a PR to the reviewed line.

## Table

| PR | year | author | draft? | bucket | title — reason | C difficulty |
|---|---|---|---|---|---|---|
| #4013 | 2026 | lvkale (current) |  | C | ck-libs/cache: CkTreeCacheManager, a process-shared tree-node cache with CkCache's protocol — New ck-libs/cache CkTreeCacheManager (process-shared tree cache); runtime-agnostic library; diff applies cleanly to the reviewed line — retarget | trivial |
| #3937 | 2026 | lvkale (current) |  | C | trace-summary: record message bytes per entry method per interval — trace-summary records message bytes per EP per interval; ck-perf compiles on both runtimes; applies cleanly to the reviewed line | trivial |
| #3910 | 2025 | ritvikrao (current) | draft | A | Charm++ support for reconverse — Original reconverse-support draft; superseded by the squash-merged core series bc819bcf2 (#3944) and follow-ups |  |
| #3904 | 2025 | ritvikrao (current) |  | C | Compile charm with CMK_LBDB_ON=0 — Lets charm compile with CMK_LBDB_ON=0 (ck.C, cklocrec.h, LBObj.C); conflicts with GPU-LB additions b17f4ac18 (#3968) in ckmigratable.h/cklocrec.h | moderate |
| #3869 | 2025 | ericjbohm (current) |  | C | add a script to calculate coverage for charm++ and converse — coverage-charm++.sh (coverage without AMPI/libs); sits beside existing coverage/coverage.sh; defaults to netlrts targets, would need reconverse target names | trivial |
| #3859 | 2024 | ericjbohm (current) |  | B | feature: support ofi-linux-arm8 — New ofi-linux-arm8 arch (OFI/CXI machine layer + craype); OFI machine layer is classic-only |  |
| #3858 | 2024 | ericjbohm (current) |  | D | allow craype for any targets (i.e., multicore) — Makes the craype option usable on any target (sets CMK_BUILD_CRAY, CMK_CRAY_NOGNI); reconverse Cray builds (Frontier, Delta) use PrgEnv gcc without craype; author must say whether reconverse-linux-* + craype is wanted |  |
| #3856 | 2024 | stevenqie (former) |  | C | added fibonacci with futures example — New futures-based Fibonacci example; Makefile hard-codes the author's laptop charmc path and must be fixed | trivial |
| #3848 | 2024 | wthrowe (former) |  | C | Mark comparison operators in ck-core const — Marks comparison operators in ck-core headers const; applies cleanly | trivial |
| #3847 | 2024 | mayantaylor (former) |  | C | Resolving hwloc cmake build issues — hwloc.m4 fix for duplicate HWLOC_SYM_* with system hwloc (issue #3843, closed without a merge); contrib/hwloc is still built for reconverse builds (cmake/hwloc.cmake) | trivial |
| #3833 | 2024 | stwhite91 (former) |  | D | Cleanup: update build/compiler examples in README — README build examples for classic targets (netlrts, ofi-crayshasta, icx); the reviewed-line README already differs (submodule/reconverse text); only the compiler-option part carries over |  |
| #3825 | 2024 | jcphill (former) |  | C | Doc: tweak +pemap offset description — Manual wording for +pemap offsets; reconverse cpuaffinity.cpp parses the same +pemap syntax | trivial |
| #3815 | 2024 | mgawan (former) |  | B | unsupported cray ftn flags being included — Cray ftn flag filtering in conv-mach-craype.sh (cray/ftn build flags) |  |
| #3807 | 2024 | mjacob1002 (former) | draft | C | Ck::IO::FileReader + Docs — Ck::IO::FileReader on top of the merged read support 4ad4095a3 (#3788); .gitignore conflict only; docs checkbox unfinished | moderate |
| #3780 | 2024 | hizv (former) |  | B | Doc: Use ++n instead of +p in charmrun examples  — Docs switch charmrun examples from +p to ++n; charmrun launch idiom is classic-only (reconverse uses ./pgm +pe N / lcrun) |  |
| #3777 | 2024 | mayantaylor (former) | draft | C | adding ckCheck to ckLocal control flow — ckCheck() in ckLocal() so uninitialized array proxies abort with a clear message (issue #3775) | trivial |
| #3768 | 2023 | ericjbohm (current) |  | B | add a gitaction for nightly build — Nightly self-hosted builds of netlrts/mpi-linux targets (might, delta); classic CI matrix. Idea overlaps plan item 17 (HPC-site runners) |  |
| #3764 | 2023 | matthiasdiener (former) |  | D | add autobuild on delta — Delta sbatch/autobuild skeleton that deletes ci.yaml and runs only placeholder echo steps; non-functional; possible seed for plan items 17/18 |  |
| #3722 | 2023 | evan-charmworks (former) | draft | B | CI: Add OFI — Adds ofi.yml CI workflow for the OFI machine layer |  |
| #3713 | 2023 | ZwFink (former) | draft | D | Disable message priorities by default — Disables message priorities by default (CMakeLists, ck.C, megatest); a policy change with no description; CMakeLists conflict |  |
| #3705 | 2023 | stwhite91 (former) |  | B | Darwin: bump Mac OS X version minimum to 10.14 — Raises -mmacosx-version-min in mpi/multicore/netlrts-darwin-x86_64 conv-mach.sh (classic arch dirs; no reconverse-darwin-x86_64) |  |
| #3676 | 2022 | lvkale (current) |  | B | Modifies converse schedeuler to prioritize NodeGroup messages — Classic conv-core getNextMessage change so nodegroup messages are polled earlier (#3674). Reconverse has its own scheduler: its slot table polls the node queues at weight 1 against 16 for the PE-local queues, so the same question needs a separate reconverse PR |  |
| #3673 | 2022 | lgtm-com bot |  | D | Add CodeQL workflow for GitHub code scanning — Bot-generated (LGTM) CodeQL workflow from 2022; would need a fresh CodeQL config for the cmake/reconverse build |  |
| #3671 | 2022 | rbuch (former) |  | B | CI: Remove manual fortran installation in spack Linux build — Edits .github/workflows/spack.yml, which no longer exists on either branch |  |
| #3649 | 2022 | mjacob1002 (former) |  | C | CkIO Example — CkIO write example (examples/charm++/ckio); library is runtime-agnostic | trivial |
| #3608 | 2022 | PathikritGhosh (former) | draft | D | Learning based LB changes — Learning-based LB draft (XGBoost model binaries in src/ck-ldb, TreeLB/MetaBalancer hooks), no description; research code |  |
| #3607 | 2022 | jszaday (former) |  | C | Add support for directing node-branch sends to PEs — Directs nodegroup sends to a chosen PE (CkEntryOptions::setNodeGroupPe, CkSendMsgNodeBranchPe); ck-core + charmxi; applies cleanly | moderate |
| #3597 | 2022 | jszaday (former) |  | C | Enable building with C++17 using `-use-new-std` — Build with C++17 via -use-new-std (CMake + charmc); reviewed line still sets CMAKE_CXX_STANDARD 11; conflicts in CMakeLists/detect-features-cxx on both branches | moderate |
| #3592 | 2022 | rbuch (former) | draft | B | Use mallinfo2 over mallinfo when available — mallinfo2 in src/conv-core/memory*.C; classic conv-core is not compiled in reconverse builds |  |
| #3589 | 2022 | ericjbohm (current) |  | C | Feature: Migratablelun — AMPI Fortran MigratableLUN tests (tests/ampi only). AMPI/TCharm are not built on reconverse yet | trivial |
| #3588 | 2022 | PathikritGhosh (former) | draft | C | Add tests useful for load balancing — Updates the kNeighbor LB example and jacobi3d test for LB experiments; applies cleanly; draft | trivial |
| #3567 | 2022 | evan-charmworks (former) | draft | C | AMPI: pieglobals updates for migration under Fortran — AMPI pieglobals Fortran-migration updates (ampi + src/util/cmitls); AMPI and cmitls are not built on reconverse yet; draft | hard |
| #3551 | 2022 | kavithachandrasekar (former) |  | D | WIP: Add DiffusionLB to loadbalancers — WIP DiffusionLB strategy, no description; author gone; overlaps Kale's diffusion/seed-balancing work (DiffusionGraphfiles), not the reconverse port |  |
| #3540 | 2021 | rbuch (former) |  | C | Add LB Position API — LB object-position API + N-dim ORB strategy; conflicts in cklocation.C/cklocrec.h/ckmigratable.h with the locmgr and GPU-LB changes | hard |
| #3507 | 2021 | minitu (former) | draft | D | Channel API in Charm++ for direct GPU-GPU communication with UCX — Draft Channel API for GPU-GPU transfers over UCX (UCX machine layer + conv-core + ck-core); the reviewed line has the LCI device path (#3955) instead; the API idea needs an owner |  |
| #3477 | 2021 | evan-charmworks (former) | draft | B | CI: Expand to run many more variants — Classic CI matrix expansion (26 workflow files, .travis.yml, .circleci) |  |
| #3436 | 2021 | matthiasdiener (former) | draft | C | cmake: add omp support — cmake: OpenMP support in buildcmake/CMakeLists (issue #3396); buildcmake conflict; draft | moderate |
| #3435 | 2021 | epmikida (former) | draft | C | CkArray: Remove duplicate array message delivery path — Removes the duplicate array-message delivery path in ck.C (_processArrayEltMsg fast path still present on the reviewed line); hot path shared by both runtimes; draft | moderate |
| #3400 | 2021 | rbuch (former) |  | C | Tests: Add AtSync test — New tests/charm++/atsync AtSync test; only tests/charm++/Makefile conflicts | trivial |
| #3394 | 2021 | evan-charmworks (former) | draft | C | Implement CmiTLS for arm64 and ppc64le — CmiTLS for arm64/ppc64le (AMPI -tlsglobals); cmitls.C is listed with classic conv-core sources and not compiled in reconverse builds, so it matters only once AMPI is ported; draft | moderate |
| #3393 | 2021 | slm960323 (former) |  | C | Fix threaded ep — Correct EP for traced threaded entry methods (CthSetEpIdx, traceResume signature); needs a reconverse-side threads.cpp change as well as the charm part | hard |
| #3392 | 2021 | ZwFink (former) |  | D | LLVM OpenMP Runtime 12.0.0 — Vendor update of LLVM OpenMP runtime to 12.0.0 (>20k-line diff); whether the OpenMP integration works on reconverse is unknown |  |
| #3347 | 2021 | ZwFink (former) |  | C | Add documentation for Converse task queues — Converse manual section on task queues; reconverse implements task queues (conv-taskq.cpp); doc/converse/manual.rst conflicts | trivial |
| #3331 | 2021 | kavithachandrasekar (former) |  | C | Avoid PUP on within node migrations — Skip PUP for within-node migrations (ckarray, cklocation, pup); conflicts with locmgr changes; relevant to the queued locmgr review | hard |
| #3329 | 2021 | matthiasdiener (former) | draft | C | cmake: honor additional libs/include dirs while configuring — cmake: honor extra lib/include dirs during configure (issue #3311); buildcmake conflict; draft | moderate |
| #3302 | 2021 | nitbhat (former) |  | B | Print a line to indicate CMA usage for ZC API — Prints a CMA-usage line in classic convcore.C |  |
| #3275 | 2021 | kavithachandrasekar (former) |  | C | Doc: Add documentation on training Metabalancer random forest model — Manual section on training the MetaBalancer random-forest model (model code exists on the reviewed line) | trivial |
| #3238 | 2021 | rbuch (former) | draft | C | Tracing: Add commSummary tracemode — commSummary trace mode (new ck-perf module); docs missing; draft | moderate |
| #3228 | 2021 | jszaday (former) |  | C | Improve Marshall Message Interface — Marshall-message interface (ckentryopts.h split, ck::make_marshall_message); touches ck-core headers | hard |
| #3193 | 2020 | nitbhat (former) |  | C | Collide: Add functionality to print the number of voxels created — Collide +printVoxCount; Collide is slated to leave the charm repo (PR #3921 context) | moderate |
| #3157 | 2020 | minitu (former) | draft | D | Add refnum with CkCallback example that hangs — Reproducer for a refnum+CkCallback hang seen on pamilrts/netlrts on Summit; untested on reconverse |  |
| #3148 | 2020 | evan-charmworks (former) |  | C | Rename pup_buffer to pup_buffer_async — Renames pup_buffer to pup_buffer_async; conflicts in pup.h/pup_util.C with reviewed-line changes | moderate |
| #3033 | 2020 | nitbhat (former) | draft | D | WIP: AMPI Bcast using ZC EM Bcast Send API — WIP AMPI Bcast via ZC broadcast API, no description; author gone |  |
| #3021 | 2020 | evan-charmworks (former) | draft | B | CI: Enable MPI Linux SMP make testp with ++ppn — Edits mpi_linux_smp.yml workflow, which no longer exists |  |
| #2836 | 2020 | stwhite91 (former) |  | B | Shrink the size of some enum types in LRTS — Shrinks LRTS enum types in gni/mpi/ucx machine layers |  |
| #2829 | 2020 | epmikida (former) | draft | D | Map refactor — Draft array-map refactor, posted only to ask which maps to keep; 44 files |  |
| #2825 | 2020 | nitbhat (former) | draft | D | WIP: Message tracking infrastructure  — WIP message-tracking infrastructure across every classic machine layer plus ck-core; no description |  |
| #2791 | 2020 | minitu (former) | draft | A | GPU-aware communication with UCX for Charm++, AMPI and Charm4Py — GPU-aware communication draft over UCX; superseded by the merged device-buffer path (342bd23b8 #2901 and successors) and the LCI device-to-device series (#3955) |  |
| #2771 | 2020 | evan-charmworks (former) | draft | B | Add support for using OpenMP workers as PEs instead of pthreads — OpenMP workers as PEs via classic machine-smp.C / machine-common-core.C |  |
| #2735 | 2020 | evan-charmworks (former) | draft | B | CI: Link the runtime as shared objects — CI shared-object link variants in classic workflows (ucx.yml, verbs.yml, .travis.yml) |  |
| #2693 | 2020 | evan-charmworks (former) | draft | B | CPU Affinity: Remove Cray-specific code — Removes Cray code from classic conv-core cpuaffinity.C (reconverse has its own cpuaffinity.cpp) |  |
| #2692 | 2020 | matthiasdiener (former) | draft | B | build: retire createlink, gather*.local, system_ln — Retires createlink/system_ln in build, QuickThreads mkfiles, win arch |  |
| #2630 | 2019 | viniciusmctf (former) |  | C | Distributed will not try to migrate tasks to itself — One-line DistributedLB fix: do not migrate an object to its own PE (issue #2629); applies cleanly | trivial |
| #2603 | 2019 | stwhite91 (former) |  | C | AMPI: add support for using liveViz from AMPI, with a wave2d example code — AMPI liveViz support + wave2d example (ampi, tcharm); AMPI/TCharm not yet on reconverse | hard |
| #2542 | 2019 | matthiasdiener (former) | draft | C | manual: add glossary — Manual glossary page; applies cleanly | trivial |
| #2434 | 2019 | epmikida (former) |  | C | Add smptest target for AMPI tests — smptest target for AMPI tests (Makefiles only) | moderate |
| #2423 | 2019 | epmikida (former) |  | D | Make it so CkArgMsg is now owned and freed by the RTS — Makes CkArgMsg owned/freed by the RTS: user-visible API change across 136 files; needs a decision |  |
| #2406 | 2019 | epmikida (former) | draft | C | More tests — Rewrites simplearrayhello, adds simplegrouptest; test-only | moderate |
| #2393 | 2019 | chin123 (former) |  | C | PICS: Fix step counters, complete implementation of more fields, and remove redundant ones — PICS step-counter fixes and field cleanup (ck-pics); applies cleanly | trivial |
| #2370 | 2019 | chin123 (former) |  | C | Change PICS configuration method to use PICS_configure — PICS_configure centralized configuration (ck-pics); applies cleanly | trivial |
| #2228 | 2019 | harshithamenon (former) |  | C | add customized reduction for CkLoop — Custom reduction for CkLoop (CkLoop still supports only built-in reduction types) | moderate |
| #2227 | 2019 | pplimport (YanhuaSun) (former) |  | B | add multiple ppn option for +ppn to handle hetergenous — Heterogeneous +ppn lists in gni/mpi/netlrts/verbs machine layers |  |
| #2225 | 2019 | pplimport (Yan-Ming Li) (former) |  | D | C++AMP direct method example — C++AMP example; depends on the defunct MCW C++AMP compiler |  |
| #2224 | 2019 | xiangni (former) |  | D | Support for out-of-core computation — Out-of-core execution research branch (29 files in ck-core) |  |
| #2222 | 2019 | xiangni (former) |  | D | Integrate ReplicaFT with master Charm — ReplicaFT integration research branch (checkpoint/pup); FT is a tombstone item in the parity plan |  |
| #2221 | 2019 | pplimport (Vipul Harsh) (former) |  | D | Support #1465 Spanning Tree implementation for scatterv — scatterv spanning-tree variant; one of three overlapping scatterv branches (#2219/#2220/#2221); needs a decision on the API |  |
| #2220 | 2019 | pplimport (Vipul Harsh) (former) |  | D | Support #1387: Node aware optimisiation for scatterv API — scatterv node-aware variant; overlaps #2219/#2221 |  |
| #2219 | 2019 | pplimport (Vipul Harsh) (former) |  | D | scatterv point to point API — scatterv point-to-point API; overlaps #2220/#2221 |  |
| #2216 | 2019 | pplimport (Steven Qiou) (former) |  | D | PAPI: initPAPI() working, better error checking — PAPI init fixes mixed with classic machine-smp.c barrier changes; needs untangling by an owner |  |
| #2211 | 2019 | pplimport (Shaoqin Lu) (former) |  | B | Change regex used for matching memory usage — Regex fix in classic conv-core memory.c |  |
| #2210 | 2019 | pplimport (Shane Neary) (former) |  | D | Feature #1713: DDT support for getting the addresses of contiguous parts of non-contiguous datatypes — DDT contiguous-block addresses, with TEMP_ AMPI entry points meant for removal |  |
| #2209 | 2019 | sbak5 (former) |  | D | Feature #1401: Converting OpenMP testsuite for the OpenMP integration. — OpenMP validation test suite (>20k lines); OpenMP-on-reconverse status unknown |  |
| #2206 | 2019 | stwhite91 (former) |  | C | Enable building charm with -fno-rtti compiler flags — Build with -fno-rtti via -DCMK_NO_RTTI (charmxi, PythonCCS); applies cleanly | trivial |
| #2201 | 2019 | stwhite91 (former) |  | C | CkArrayMap: change ReadFileMap file format to simple chare-to-pe mapping — ReadFileMap reads a simple chare-to-PE file (cklocation.C); changes a file format | moderate |
| #2200 | 2019 | stwhite91 (former) |  | D | AMPI cleanup recv code path by normalizing argument ordering — AMPI recv-path argument reordering plus a classic convcore.c hunk; ampi.C has diverged since 2018 |  |
| #2199 | 2019 | stwhite91 (former) |  | D | AMPI cleanup send code path by normalizing argument ordering — AMPI send-path argument reordering plus a classic convcore.c hunk; ampi.C has diverged since 2018 |  |
| #2194 | 2019 | stwhite91 (former) |  | D | AMPI: add pt2pt-based algorithms for short and long Allreduce msgs — AMPI pt2pt Allreduce, but the branch also carries unrelated zerocopy examples (51 files); needs untangling |  |
| #2186 | 2019 | stwhite91 (former) |  | C | AMPI cleanup #1346: remove AMPIMSGLOG support — Removes AMPIMSGLOG support (still present on the reviewed line) | moderate |
| #2184 | 2019 | stwhite91 (former) |  | C | AMPI #1158: implement Scatter(v) as a [nokeep] Bcast — AMPI Scatter(v) as a [nokeep] broadcast; related to the nokeep work; AMPI not yet on reconverse | hard |
| #2182 | 2019 | pplimport (Ralf Gunter) (former) |  | D | charmxi #921: dummy group commit — charmxi #921 "dummy group commit" (charmrun + xi-Entry) |  |
| #2181 | 2019 | PhilMiller (former) |  | B | SAMPLE: Convert home-grown skt_sendV to standard writev — netlrts machine.c: writev instead of skt_sendV |  |
| #2180 | 2019 | PhilMiller (former) |  | C | Bug #902: Record originating PE and event number for array element insertions — Record originating PE/event for array insertions; CkArray::insertElement is still [inline]; regenerates grammar tables | hard |
| #2179 | 2019 | PhilMiller (former) |  | D | TRAM: Reduce default flush interval to 1ms — Mislabeled (not a TRAM change): stacks the #2177 broadcast change plus a trace hunk; the TRAM ckNew code it edits is gone from xi-Entry.C |  |
| #2177 | 2019 | stwhite91 (former) |  | D | Array Broadcasts: Don't serialize via reflection off PE 0 — Array broadcasts no longer serialized through PE 0; changes broadcast ordering; needs a design decision |  |
| #2176 | 2019 | PhilMiller (former) |  | A | Refactor (node)group message delivery to not go through an extraneous function — Group/nodegroup delivery no longer goes through _processForChareMsg: equivalent in 8f18a7836; _processForBocMsg calls _invokeEntry directly |  |
| #2175 | 2019 | PhilMiller (former) |  | A | Respect group construction dependence in NodeGroup constructor invocation — Group-dependence check in nodegroup constructor: equivalent in d24d191d9 (isGroupDepUnsatisfied in _processNodeBocInitMsg) |  |
| #2173 | 2019 | PhilMiller (former) |  | D | charmxi #921: actually emit definition of virtual method FooClosure::preserve() — charmxi #921 SDAG const-correctness series (with #2169-#2171), 2016 experiments |  |
| #2172 | 2019 | PhilMiller (former) |  | D | NOMERGE Drop SDAG tests that use 'forall' construct, to avoid generated code error — NOMERGE: drops SDAG forall tests to work around codegen |  |
| #2171 | 2019 | PhilMiller (former) |  | D | charmxi #921: push through const-correct parameter types for arrays — charmxi #921 series: const-correct array params |  |
| #2170 | 2019 | PhilMiller (former) |  | D | charmxi #921: carry const-ness through SDAG in [inline] methods - works for const& and value — charmxi #921 series: const-ness through SDAG [inline] |  |
| #2169 | 2019 | PhilMiller (former) |  | D | experimental scheme for avoiding overhead in inline calls by — Experimental inline-call indirection scheme (charmxi #921 series) |  |
| #2168 | 2019 | PhilMiller (former) |  | C | Delete immediate message support in QD to eliminate a use of CpvAccessOther — Drops immediate-message QD path (CpvAccessOther in ck.h, still present); CHANGES hunk is stale and should be dropped | moderate |
| #2167 | 2019 | PhilMiller (former) |  | D | NOMERGE: Pass marshalled entry method parameters by rvalue reference — NOMERGE: marshalled params by rvalue reference (23 files) |  |
| #2166 | 2019 | PhilMiller (former) |  | C | MPI Interop: add routine to run Charm++ scheduler until QD — MPI interop: run the scheduler until QD; MPI interop is untested on reconverse | moderate |
| #2165 | 2019 | PhilMiller (former) |  | A | NOMERGE Squash location manager cleanup and 64-bit ID work for testing — NOMERGE squash of locmgr cleanup and 64-bit IDs; that work landed as 71a0f8961 and successors |  |
| #2163 | 2019 | PhilMiller (former) |  | C | Allow message-array alignment to minimum necessary boundary — Message-array alignment to the minimum needed boundary (ck-core, charmxi, configure.in); stale autoconf parts | hard |
| #2159 | 2019 | nitbhat (former) |  | B | Testcase for Bug #1671: Example to reproduce Verbs mempool bug — Reproducer for a Verbs mempool bug |  |
| #2156 | 2019 | nitbhat (former) |  | B | POC to remove comm thread from OFI-smp build — POC removing the comm thread from OFI-smp (classic SMP comm thread) |  |
| #2155 | 2019 | nitbhat (former) |  | C | ZC Direct API: Allow users to not invoke callbacks for inline transfers — ZC Direct API option to skip callbacks for inline transfers (CkNcpyCallbackMode); only CK_EP_INLINE landed | hard |
| #2154 | 2019 | nitbhat (former) |  | D | Apply clang-format on converse examples and tests — clang-format of converse examples/tests (44 files); style churn with no owner |  |
| #2153 | 2019 | nitbhat (former) |  | B | Feature #1931: Direct API for pamilrts-linux-ppc64le — Direct API for pamilrts |  |
| #2152 | 2019 | nitbhat (former) |  | B | Bug #1708: Fix multi-node application hangs on mpi-crayxc and mpi-crayxe — mpi-crayxc/crayxe hugepage checks in charmc |  |
| #2151 | 2019 | nitbhat (former) |  | A | build script: Pick libraries from lib64 sub-dir inside "--basedir" dir — Link with <basedir>/lib64: equivalent in e1a85fbae (#3532, buildcmake) |  |
| #2150 | 2019 | nitbhat (former) |  | B | Fix for a weird bug that Sameer found in priority messages code while running NAMD — NAMD-on-PAMI priority bug fix across pami layer + classic convcore (plus a CkLoop hunk) |  |
| #2149 | 2019 | juanjgalvez (former) |  | D | NO MERGE YET - Extend TopoManager to provide ordered list. — NO MERGE YET: TopoManager ordered-node partitions (feedback requested) |  |
| #2148 | 2019 | nikhil-jain (former) |  | B | NO MERGE: Explicit messages for exit in SMP mode. — NO MERGE: explicit exit messages to the SMP comm thread |  |
| #2147 | 2019 | nikhil-jain (former) |  | D | NO_MERGE: Drop inline chare array creation — NO_MERGE: showcase for dropping inline array creation |  |
| #2144 | 2019 | mprobson (former) |  | D | accel: Implement  parsing for [accel] targets in .ci — [accel] entry-method parsing (67 files); accel research line |  |
| #2140 | 2019 | matthiasdiener (former) |  | D | doc: split charm manual into several files — Splits the Charm++ manual into files; manual.rst has diverged; a redo, not a rebase, if still wanted |  |
| #2138 | 2019 | matthiasdiener (former) |  | B | cleanup: remove duplicate CmiBarrier() calls — Removes duplicate CmiBarrier calls, mostly in gni/mpi/pami machine layers and classic convcore |  |
| #2137 | 2019 | matthiasdiener (former) |  | D | Cleanup: match ci filename to module name — Renames .ci files to match module names (46 files); churn with no owner |  |
| #2135 | 2019 | matthiasdiener (former) |  | C | AMPI: Update MPI version to 3.1 — Raise AMPI MPI_VERSION to 3.1 (still 2.2 on the reviewed line) | trivial |
| #2133 | 2019 | karthiksenthil (former) |  | C | Feature #969: AMPI support for inter-communicator (i)gather collectives. — AMPI intercomm (i)gather; intercomm bcast/scatter exist on the reviewed line, gather does not | hard |
| #2132 | 2019 | karthiksenthil (former) |  | C | Feature #969 : AMPI support for inter-communicator collectives. — AMPI intercomm collectives (barrier, gather(v)); partly superseded (bcast/scatter present) | hard |
| #2131 | 2019 | pplimport (Justin Szaday) (former) |  | D | Added support for attribute arguments to charmxi. — charmxi attribute arguments (2017); needs an owner |  |
| #2130 | 2019 | pplimport (Justin Szaday) (former) |  | D | Whitespace cleanup for Change 2792. — Whitespace cleanup for an old Gerrit change |  |
| #2129 | 2019 | pplimport (Justin Szaday) (former) |  | B | Added (primitive) support for to use OpenMP threads as Charm workers. — OpenMP threads as workers via machine-smp.c + user-driven interop (unsupported on reconverse) |  |
| #2127 | 2019 | juanjgalvez (former) |  | C | Remove location entries if chare migrates and entry is not used — Remove unused location-table entries after migration (cklocation); for the locmgr review | hard |
| #2125 | 2019 | juanjgalvez (former) |  | A | NO MERGE YET: Added CharmPy support for sections using ckmulticast — Charm4py sections via ckmulticast: landed as a4282772b and 15a770242 |  |
| #2124 | 2019 | juanjgalvez (former) |  | C | TopoManager: Added API to query physical node info — TopoManager physical-node query API; TopoManager is built on reconverse | moderate |
| #2123 | 2019 | lifflander (former) |  | C | charmc: external overriding of compiler variables (pkg manager) — charmc: external override of compiler variables for package managers | moderate |
| #2122 | 2019 | stwhite91 (former) |  | C | make: fix install script to work properly — Fix the install target in src/scripts/Makefile (install loop still the old form) | trivial |
| #2116 | 2019 | harshithamenon (former) |  | D | Add mirror for ckarray so that requests can be handled by the mirror — Array mirror research (13 ck-core files) |  |
| #2113 | 2019 | harshithamenon (former) |  | C | Fix ckloop race condition for non-sync mode — CkLoop non-sync race fix (numRunning); CkLoop still uses the finishFlag test; CkLoop is used by NAMD | moderate |
| #2112 | 2019 | harshithamenon (former) |  | D | Node-level cache for SMP version of Charm. — Old node-level SMP cache for CkCache; same goal as #4013, which should replace it |  |
| #2111 | 2019 | mprobson (former) |  | D | Add accelerated entry methods — Accelerated entry methods (David Kunzman thesis line), 67 files |  |
| #2101 | 2019 | evan-charmworks (former) | draft | B | Add IPv6 support — IPv6 in netlrts/verbs/CCS (classic conv-ccs is not compiled in reconverse builds) |  |
| #2098 | 2019 | evan-charmworks (former) |  | B | Shrink/Expand: Fix the calculation of oldPEs to work correctly with +n specified instead of +p — charmrun shrink/expand oldPEs fix |  |
| #2091 | 2019 | pplimport (Edward Hutter) (former) |  | D | RMA Support for DDT — RMA for DDT; ampiOneSided.C already branches on isContig(); overlap unclear |  |
| #2090 | 2019 | pplimport (Edward Hutter) (former) |  | D | AMPI #1258: derived datatype support for RMA routines — DDT for RMA routines (older sibling of #2091); overlap unclear |  |
| #2088 | 2019 | pplimport (Edward Hutter) (former) |  | C | AMPI: add example particle code — AMPI particle test code (tests/ampi only) | trivial |
| #2087 | 2019 | stwhite91 (former) |  | A | AMPI #1105: add MPI-3 large count support to AMPI and DDT — MPI-3 large-count routines: landed separately (e.g. 5d959a5ea, f09f4173d) |  |
| #2086 | 2019 | pplimport (Edward Hutter) (former) |  | A | AMPI #947: Added support for Subarray datatype to CkDDT and AMPI. — Subarray datatype: landed as 548c8eb7f |  |
| #2085 | 2019 | pplimport (Orion Lawlor) (former) |  | A | Replace the memPool lists with standard "mempool.h".  This makes one big cudaMallocHost, and vastly simplifies memory handling in the hybrid API. — HAPI host-memory pool: HAPI rewritten with buddy allocator (2383a7185 #2896); files touched no longer exist |  |
| #2084 | 2019 | bilgeacun (former) |  | B | OFI: Configuration fixes for OFI on Pacini cluster. — OFI configuration in netlrts (Pacini cluster) |  |
| #2083 | 2019 | evan-charmworks (former) |  | D | Add option to use libperf (incomplete, do not merge) — libperf option, marked incomplete/do not merge |  |
| #2082 | 2019 | pplimport (Steven Qiou) (former) |  | C | PAPI: add command line interface — PAPI event file via +papi command-line option (ck-perf); .tex docs need converting to .rst | moderate |
| #2080 | 2019 | evan-charmworks (former) |  | B | Use daemon to run multiple tests with one set of SSH connections to nodes — charmrun daemon to reuse ssh connections across tests |  |

## C items in priority order

Tier 1: small, current, or directly useful to the reviewed line now.

1. **#4013** (lvkale) CkTreeCacheManager: new library work that belongs only on the reviewed line; retarget before the freeze.
2. **#3937** (lvkale) trace-summary bytes: Projections volume data for summary traces; ck-perf builds on both runtimes; retarget.
3. **#3904** (ritvikrao) CMK_LBDB_ON=0 build: a current author; fixes a build configuration on the shared ck-core/ck-ldb code; rebase against the GPU-LB additions (#3968).
4. **#3435** duplicate array delivery path: simplifies the hot `_processArrayEltMsg` path that the nokeep audit just worked on; ties into the locmgr review.
5. **#2113** CkLoop non-sync race: a correctness fix in a library NAMD uses; CkLoop runs unchanged on reconverse.
6. **#2630** DistributedLB self-migration: a one-line correctness fix for a stall (issue #2629).
7. **#3847** hwloc.m4 duplicate symbols: contrib/hwloc still builds for reconverse, and the fix was confirmed by the reporter of #3843.
8. **#3777** ckCheck in ckLocal: one-line clearer abort for uninitialized proxies.
9. **#3848** const comparison operators: header correctness; applies cleanly.
10. **#3597** C++17 build option: the reviewed line still sets CMAKE_CXX_STANDARD 11 while the darwin arch already forces gnu++17; a coherent C++17 baseline is needed for the new line.
11. **#3607** nodegroup sends to a chosen PE: a user-requested API (Nils Deppe); runtime-agnostic.
12. **#2168** drop immediate-message QD path: removes a CpvAccessOther use in ck.h that reconverse does not need.

Tier 2: tests, examples and docs. They are cheap and grow the reviewed-line test tier (plan item 13a).

13. **#3400** AtSync test: a missing test for a core LB path.
14. **#2406** simplearrayhello rewrite + simplegrouptest: adds group coverage.
15. **#3588** LB kNeighbor/jacobi3d updates: LB experiment drivers.
16. **#3649** CkIO write example: no CkIO example exists on the reviewed line.
17. **#3807** CkIO FileReader: completes the merged read support (#3788); needs its docs finished.
18. **#3856** futures Fibonacci example: fix the hard-coded charmc path in its Makefile.
19. **#3825** +pemap offset wording: reconverse parses the same syntax.
20. **#3347** Converse task-queue docs: reconverse implements task queues.
21. **#3275** MetaBalancer random-forest docs: the model code is present.
22. **#2542** glossary: applies cleanly.
23. **#3869** (ericjbohm) coverage-charm++.sh: complements the coverage campaign; it needs reconverse target names.

Tier 3: moderate features in shared code.

24. **#3540** LB position API + ORB: a useful LB feature, but it conflicts with the locmgr and GPU-LB changes.
25. **#3331** skip PUP on within-node migration: performance; belongs in the locmgr review.
26. **#2127** prune unused location entries: belongs in the locmgr review.
27. **#3393** correct EP for traced threaded entries: needs a companion reconverse threads.cpp change; relevant to record-replay item 19 (threaded-entry validation).
28. **#3228** marshall-message interface: groundwork for ck::callback; hard rebase in ck-core headers.
29. **#3148** pup_buffer_async rename: an API rename; conflicts in pup.h.
30. **#3238** commSummary trace mode: new tracer; needs docs.
31. **#2082** PAPI +papi event file: tracing usability; docs need rst conversion.
32. **#2393** and **#2370**: PICS fixes; apply cleanly; ck-pics builds under TRACING.
33. **#2228** CkLoop custom reductions.
34. **#2124** TopoManager physical-node API.
35. **#2201** ReadFileMap format: a file-format change; needs agreement.
36. **#2166** MPI interop run-until-QD: MPI interop is untested on reconverse (open item).
37. **#2155** ZC inline-callback option: only CK_EP_INLINE landed.
38. **#2163** message-array alignment: stale autoconf parts.
39. **#2180** array-insertion trace origin: requires regenerating the charmxi grammar tables.
40. **#3193** Collide voxel count: Collide is slated to leave the charm repo.
41. Build and install: **#3436** cmake OpenMP, **#3329** cmake extra lib dirs, **#2123** charmc compiler override, **#2122** install target, **#2206** -fno-rtti.

Tier 4: AMPI. It is still wanted, but it cannot run on reconverse until TCharm/AMPI are ported.

42. **#2135** MPI_VERSION 3.1, **#2088** particle test, **#3589** (ericjbohm) MigratableLUN tests, **#2434** smptest target, **#2186** remove AMPIMSGLOG, **#2184** Scatter as [nokeep] Bcast, **#2132** and **#2133** intercomm collectives (bcast/scatter already present), **#2603** liveViz, **#3567** pieglobals Fortran, **#3394** CmiTLS arm64/ppc64le.

## Notes on specific A citations

- #3910: bc819bcf2 (#3944).
- #2791: 342bd23b8 (#2901) plus #3955.
- #2176: 8f18a7836.
- #2175: d24d191d9.
- #2165: 71a0f8961.
- #2151: e1a85fbae (#3532).
- #2125: a4282772b and 15a770242.
- #2087: 5d959a5ea and f09f4173d.
- #2086: 548c8eb7f.
- #2085: 2383a7185 (#2896).

Each of these is an equivalent change, not the same patch. None of the PR diffs reverse-applies.
