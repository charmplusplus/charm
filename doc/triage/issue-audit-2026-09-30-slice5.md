# Issue audit, slice 5 (charmplusplus/charm, 94 issues, created 2021-09 to 2026-09)

Audited 2026-09-30 against `origin/reviewed-with-reconverse` at ebdc49c66 (reconverse submodule b30ad31). Read-only: issue bodies and last comments via `gh`, code via `git show`/`git grep` on the branch, and a local run of `tests/charm++/io_read` (#3872). The slice file also contained 2021 issues (#3469-#3541), which are included.

Buckets: K keep (defect or limitation on the reviewed line), F fixed/superseded, O obsolete on the reviewed line, A AMPI/TCharm port list, I idea/wishlist, D needs reporter or design call. `*` = critical K.

| issue | year | labels | bucket | critical? | reason / evidence |
|---|---|---|---|---|---|
| #3469 multicore-darwin-arm8 exhibits poor scaling | 2021 | MacOS,ARM | O |  | Poor within-node scaling of classic `multicore-darwin-arm8` (M1, SpECTRE); classic multicore layer is not built; reconverse-darwin-arm8 replaces it (re-measure there if SpECTRE reports again). |
| #3471 MetisLB symbols conflict with AMPI applications using METIS | 2021 | AMPI,load balancing | A |  | Embedded METIS (MetisLB) symbols clash with AMPI apps that link their own METIS; needs a symbol prefix like hwloc's `cmi_`. Also hits Charm++ apps using METIS, so carry into the AMPI port list with that note. |
| #3479 Group section failure | 2021 | - | D |  | Hang with two CkMulticast sections rooted on PE0 and PE2 (student debug branch `debug_groupsection`); no reproducer on the reviewed line and no triage; could be user code. Needs a rerun of the branch's test. |
| #3483 Add object position API for LB | 2021 | - | I |  | API for chares to register a position for geometric LBs; feature request. |
| #3486 Build Charm4Py with other networking layers | 2021 | - | D |  | Charm4Py `libcharm.so` only links netlrts/mpi libs. The netlayer part is obsolete, but whether the `charm4py` target (CMakeLists.txt:1112, links `converse` only) pulls in libreconverse/LCI correctly is untested. |
| #3488 Add GPU/node topology query interface | 2021 | smp,GPU support | I |  | GPU/node topology query interface (NVLink via hwloc) for placement; feature. |
| #3491 user-driven-interop fails at exit on mpi-linux-* | 2021 | mpi interoperability | O |  | `user-driven-interop` exit failure on `mpi-linux-*` (Travis); MPI machine layer not built, and user-driven interop is unsupported on reconverse anyway. |
| #3501 entry method called from inside CkExit | 2021 | - | K |  | Entry methods still run after a non-PE0 `CkExit`: `CkExit` (init.C:1131) sends StartExitMsg to PE 0 and enters `CsdScheduler(-1)`; `_discardHandler` is installed only later by `_exitHandler`. Violates the manual's guarantee. |
| #3519 NAMD hangs with mpi-smp build on DGX A100 | 2021 | Bug | O |  | NAMD hang in `CmiCheckAffinity` with `ucx-linux-x86_64 smp` on DGX A100; classic UCX layer. |
| #3531 Support Hierarchical Assignment in TreeLB | 2021 | load balancing | I |  | TreeLB hierarchical assignment (root assigns to nodes, nodes to PEs); enhancement. |
| #3539 Support exported CMake targets | 2021 | CMake | I |  | Exported CMake targets / `find_package(Charm)` without charmc (SpECTRE, Enzo-E ask); CMake is now the only build system, so this is a good wishlist item. |
| #3541 Support messages longer than 32 bits | 2021 | - | I |  | Messages longer than 32-bit size; envelope size is still 32-bit. Feature (few users need >4 GB messages). |
| #3546 Charm++ cannot configure due to faulty ${PYTHON_VERSION} | 2022 | Buildold | O |  | `PYTHON_VERSION` newline bug in autoconf `configure` (buildold); the CMake build computes `CMK_PYTHON_VERSION` itself (CMakeLists.txt:352). |
| #3556 CmiCheckAffinity causes hang with UCX build on Summit | 2022 | - | O |  | `CmiCheckAffinity` hang on Summit with UCX/PAMILRTS; classic layers, Summit retired. |
| #3561 MSA Examples Failing | 2022 | - | K |  | MSA examples `matmul` (segfault in `MSA_CacheGroup::accessPage`) and `moldyn` (hang) broken; MSA still built (ck-libs/CMakeLists.txt:14); only #3562 touched it since. |
| #3564 Build switches compiler in "charmconfig.out" on MacOS bigSur | 2022 | - | O |  | macOS gcc-11 build mixing compilers in `charmconfig.out`; autoconf-era build. |
| #3575 Default PMIx included in IBM Spectrum MPI degrades GPU-aware communication performance on OLCF Summit | 2022 | GPU support,UCX,Performance | O |  | Summit Spectrum-MPI PMIx degrading UCX GPU-aware comm; classic UCX, Summit retired. |
| #3587 Fail build if with-production and enable-replay both requested? | 2022 | - | F |  | Asks to fail when production and replay are combined. On the reviewed line replay is tied to tracing, not production, and CMake warns when `REPLAY` is set without `TRACING` (CMakeLists.txt:259-261). |
| #3598 Using +ppn instead of ++ppn should not be silently ignored by netlrts layer | 2022 | - | O |  | `+ppn` silently ignored by netlrts (needs `++ppn`); charmrun/netlrts gone. (Worth checking that reconverse rejects a stray `+ppn` rather than ignoring it.) |
| #3610 Study performance impact of avoiding doing array broadcasts via PE0 | 2022 | enhancement | I |  | Array broadcasts serialized through PE 0; proposal to get a sequence number and broadcast from the origin. Performance study. |
| #3611 Study allowing out-of-order delivery of array broadcasts | 2022 | enhancement | I |  | Out-of-order delivery of array broadcasts (epoch + bitset); depends on #3610. |
| #3613 Error when using Projections with shrink-expand | 2022 | Bug,tracing,shrink-expand | K |  | Projections tracing plus shrink/expand aborts at startup: `CkAbort("registerExitFn is called when shrink-expand is enabled!")` still at init.C:1886, and trace-projections registers an exit fn. |
| #3617 Performance boost when using `+skip_cpu_topology` | 2022 | Performance | D |  | 3-6% NAMD gain with `+skip_cpu_topology` on Cray EX (UCX/MPI); reconverse has its own topology code (conv-topology.cpp), so whether the cost carries over is unknown and needs a measurement. |
| #3619 Absolute paths in symbolic links can break Charm++ builds | 2022 | - | O |  | Absolute symlinks in the build `tmp/`, found with the old build system and not reproduced. |
| #3621 Add additional template based tests | 2022 | Build & test automation | I |  | More tests for templated chares/EPs; test coverage. |
| #3622 Add C++11 support for `CMK_NATIVE_COMPILER` to CMake build | 2022 | enhancement,CMake | O |  | C++11 for `CMK_NATIVE_COMPILER` in CMake; only affects cross-compiled classic layers. CMake builds charmxi with the project compiler at C++11+ (CMakeLists.txt:46). |
| #3634 Improved parameter marshalling for [reductiontarget] entry methods | 2022 | enhancement,charmxi | I |  | `[reductiontarget]` receiving `std::vector` directly; charmxi enhancement. |
| #3645 Does the cmake build work with clang? | 2022 | CMake,Fortran | F |  | charmc `eval -I` failure when CMake blanks the Fortran compiler; option to disable Fortran merged (#3646). |
| #3655 [Feature request] Limit number of inline calls to avoid blowing stack | 2022 | enhancement | I |  | Limit nested inline entry-method calls to avoid stack overflow (SpECTRE patch); feature. |
| #3662 Add dynamic insertion tests | 2022 | enhancement,Build & test automation | F |  | Dynamic insertion tests added in 54f4783a8 (`dynamic_insertion`, `dynamic_insertion_deletion`, amr_1d tests; tests/charm++/Makefile:32-33). |
| #3670 Add `ckDestroy` to generated array proxy | 2022 | - | I |  | Generated `CProxy_Array::ckDestroy()` wrapping `CkArray::ckDestroy`; charmxi convenience. |
| #3674 Nodegroup messages are too low a priority | 2022 | - | K |  | Nodegroup messages starved by local work. Reconverse's default registered scheduler weights the node queue 1 against 16 for the self and thread queues (scheduler_registered.cpp:124-128), so the imbalance remains. Classic PR #3676 is still open. |
| #3680 Running NAMD built with gcc on Rocky 8 against Charm++ versions >= v7.0.0 built on CentOS 7.x causes seg fault | 2022 | - | O |  | Not a defect: NAMD mixed pieces of Charm++ 6.x and 7.x, which have incompatible LB ABIs. |
| #3686 Provide compile-time warning for use of `atomic` in SDAG | 2023 | cleanup,charmxi | I |  | Compile-time warning for deprecated SDAG `atomic`; xi-scan.l:134 still maps it silently to SERIAL. |
| #3731 Pool small messages inside the runtime | 2023 | enhancement,Performance,Converse | I |  | Pool small (marshalled) messages in the runtime; performance idea. |
| #3762 Bad memory accesses in springCleaning() | 2023 | Bug | K | * | Use-after-free in `CkArray::staticSpringCleaning` (ChaNGa+CkIO); #3765 only works around it for CkIO WriteSession. Likely root cause (found by reading, shared by classic and reconverse conv-conds): `CcdCallOnCondition` returns a vector index, which becomes wrong once the list fires and is compacted, so `~CkArray`'s `CcdCancelCallOnCondition(CcdPERIODIC_1minute, springCleaningCcd)` erases the wrong or out-of-range slot and leaves the dead array's callback live. |
| #3770 GPU build failing on delta | 2023 | - | F |  | CMake CUDA link failure (MPI+CUDA) on Delta. Superseded: reconverse CUDA builds are built and run on Delta (site-run tier, #3982/#3983 jobs). |
| #3775 Error message from using uninitialized proxy | 2023 | - | I |  | Clearer error for use of an uninitialized proxy (currently "Group ID is zero-- invalid!"); PR #3777 open. |
| #3778 Charm++ breaks the RAII of C++ | 2024 | - | I |  | Chare destructors never run at exit (RAII); design limitation, a candidate for an opt-in teardown. |
| #3783 Move UIUC PPL wiki somewhere else | 2024 | - | O |  | Move the UIUC PPL wiki (retired 2024); organizational, and nothing in the tree references wiki.illinois.edu. |
| #3789 Build fails with strict-aliasing violations | 2024 | - | K |  | `-Werror=strict-aliasing` LTO build failures. Most hits are classic conv-core/QuickThreads, but shared `src/util/pup_toNetwork.h:117` still does `*(void **)&i`. Minor. |
| #3790 Fail to build with address sanitizer | 2024 | - | K |  | `-fsanitize=address` builds fail because LeakSanitizer flags charmxi's leaks (`xi::Entry::genClosure`) at build time. Minor; `ASAN_OPTIONS=detect_leaks=0` is the workaround. |
| #3794 Support XPMEM interprocess communication within node when SMP is used within process | 2024 | machine layers,smp,Performance | O |  | XPMEM plus SMP in classic machine layers; reconverse has its own `cmixpmem.cpp`. |
| #3803 Unable to clone charm fft | 2024 | - | K |  | Gerrit URLs are dead (503) but still cited in docs: doc/ampi/03-using.rst:544, doc/ampi/05-examples.rst:373, doc/charisma, doc/debugger/manual.rst:58. FFT lib and OpenAtom data unreachable. Docs rot. |
| #3812 Fast lock-free queue for nodegroup messages | 2024 | - | F |  | Lock-free nodegroup queue: reconverse's node queue is moodycamel `ConcurrentQueue` (queue.h:51, `RECONVERSE_ATOMIC_QUEUE` ON by default), the queue proposed here. |
| #3818 Segfault when built with --enable-tracing-commthread and isomalloc | 2024 | Bug | O |  | Segfault with `--enable-tracing-commthread` + isomalloc on verbs smp; there is no comm thread on reconverse. |
| #3826 Bug with reductions when ckJustMigrated is implemented in user code | 2024 | Bug | I |  | Overriding `ckJustMigrated` without calling `ArrayElement::ckJustMigrated` breaks reduction counts and hangs. Documented user contract (olawlor); API hardening (split system plumbing from the user hook) is optional. |
| #3831 CmiPushPE doesn't account for message injection from outside the Charm++ runtime | 2024 | - | F |  | `CmiPushPE` from foreign (CUDA) threads. On reconverse `Cmi_myrank` is `thread_local` initialised to -1 (convcore.cpp:89), so a foreign thread always takes the shared MPSC queue (convcore.cpp:606-610). |
| #3832 curPeEvent reference causes SEGFAULT | 2024 | tracing | K |  | `[local]`/`[inline]` EPs compiled at -O0 against a production (non-tracing) build dereference uninitialised `CpvAccess(curPeEvent)`: xi-Entry.C:565/902 emit it unconditionally, and it is only initialised in trace-common.C. Workaround `[notrace]`. |
| #3840 mpi-linux-arm8 build crashes on  > 128 cores for hello/3darray | 2024 | - | O |  | `mpi-linux-arm8` crash above 128 cores on Vista; MPI layer. |
| #3842 Build fails due to CMake 3.28 identifying Cray PE as 'CrayClang' | 2024 | - | K |  | CMake 3.28+ reports Cray PE as `CrayClang`; CMakeLists.txt:49-82 has no branch for it and hits `FATAL_ERROR "Unknown compiler"`. Blocks direct Cray PE builds; the patch is in the issue. |
| #3845 Add test for NETWORK variable | 2024 | - | K |  | `NETWORK` unset crashes CMake: it is dereferenced at CMakeLists.txt:39 before its cache default at :110, and that default is "netlrts", wrong for this line. |
| #3846 Relation problem during compilation | 2024 | - | K |  | Direct-CMake shared build: embedded hwloc is built without PIC unless charm's own `BUILD_SHARED` is set (cmake/hwloc.cmake:4); `CMAKE_POSITION_INDEPENDENT_CODE`/`BUILD_SHARED_LIBS` ignored. Same theme as #3845/#3539. |
| #3850 Dramatic slowdown ( 50x typical) when --with-production used with *-linux-arm8 on multi-node NVIDIA Grace Hopper runs | 2024 | - | D |  | 50x slowdown with `--with-production` multi-node on Grace Hopper (mpi and netlrts, periodic ~30 s stalls). Root cause unknown; both layers are classic. Needs a rerun on reconverse-linux-arm8. |
| #3853 Implement +maffinity support using hwloc | 2024 | - | I |  | `+maffinity` via hwloc membind instead of libnuma; feature. |
| #3854 Using --install-prefix gives a conv-mach-opt.sh which still references the source directory | 2024 | - | K |  | `--install-prefix` installs a `conv-mach-opt.sh` that sources `${CMAKE_BINARY_DIR}/include/cc-*.sh` by absolute build path (CMakeLists.txt:1169), so the install breaks once the build tree is deleted (Spack). |
| #3860 Possible message drop when receiving large number of messages | 2024 | - | D |  | SpECTRE: a message dropped after ~2.2e9 (about 2^31) messages to one element, QD silent, on `mpi-linux-x86_64-smp`. QD counters are 64-bit (qd.h:15-17), so the overflow site is unknown and may be classic MPI. Would be critical if it reproduces; run the attached MessageDrop reproducer on reconverse. |
| #3861 Is there a way to have access to a Charm++ version or development branch that has HIP support for AMD GPU use? | 2024 | - | F |  | HIP/AMD support: the reviewed line has HAPI portable core with HIP (#3951) and runs on Frontier MI250X. |
| #3862 ckout does not work with `long double` | 2024 | - | K |  | `ckout << long double` is an ambiguous overload (no `long double` operator in ckstream.h); trivial API gap. |
| #3868 netlrts smp hangs at startup with some process/thread count configurations | 2025 | - | O |  | netlrts SMP startup hang for some process/thread counts (bisected to b7b126dc6); netlrts/charmrun gone. |
| #3872 io_read test broken on MacOS 13 | 2025 | Bug,Build & test automation,MacOS,CI | K | * | io_read (CkIO read) crashes on macOS. Reproduced today on BOTH runtimes, local builds, arm64 Mac: reconverse `+pe 1/2/4` and classic netlrts `+p1` both SIGSEGV after all reads verify, in `CkLocRec::invokeEntry` <- `CkArrayBroadcaster::attemptDelivery` <- `CkArray::recvBroadcast`, i.e. a broadcast reaching an element with no location record. io_read is not in the reconverse CI tier. |
| #3877 CMK_CXI not defined on non-Slingshot OFI build | 2025 | - | O |  | `CMK_CXI` undefined on a non-Slingshot OFI build; classic OFI layer. |
| #3879 Shrink/Expand status | 2025 | - | D |  | User asks which branch supports shrink/expand (main example broken; `shrinkexpand-fix` expands but won't shrink). The reviewed line has checkpoint-based shrink/expand; needs a status answer and docs, not a code verdict. |
| #3888 Running with Flux Framework | 2025 | - | I |  | Launch under Flux. Reconverse bootstraps through LCI PMI, so Flux's PMI may already work; document or test. |
| #3892 intermittent segfaults when running allGather | 2025 | Bug | D |  | allGather library segfaults at 128 PEs on one node (from #3886 review); maintainer asked whether it reproduces on the merged code, no answer. Needs a run. |
| #3901 ifx / ifort confusion | 2025 | - | O |  | `ifx` requested but `ifort` used, via `mpi-linux-x86_64/conv-mach.sh:40` (classic arch). The same class of bug on this line is #3971. |
| #3915 Shrink/expand questions | 2025 | - | D |  | Shrink/expand usage questions (expand from +p2 fails with a socket error, CCS client, AMPI malleability); classic charmrun/CCS path. The reviewed line's checkpoint-based shrink/expand needs its own answer. |
| #3920 Support for Windows on ARM (ARM64) | 2026 | - | D |  | Windows on ARM64. The reviewed line has no Windows reconverse arch at all, so the question is whether Windows is supported on the new line. |
| #3933 Projections: .sts provenance is incomplete (PROJECTIONS_ID empty; COMMANDLINE misses pre-snapshot args; no app version / job id / environment) | 2026 | - | K |  | `.sts` provenance incomplete: `PROJECTIONS_ID` is always the empty string (trace-projections.C:412); COMMANDLINE is a post-strip snapshot. |
| #3934 trace-summary: .sum idle/utilization and .sumd per-EP times disagree in +sumDetail traces | 2026 | - | K |  | trace-summary `.sum` idle/utilization and `.sumd` per-EP times come from two unreconciled accumulators (`SumLogPool::add` vs `updateSummaryDetail`) and disagree, measured on a reconverse trace; no fix merged. |
| #3935 ARM64 with Infiniband is not supported | 2026 | - | F |  | `verbs-linux-arm8 smp` unsupported (ARM64 + InfiniBand). Superseded: `reconverse-linux-arm8` over LCI (ibverbs/OFI backends) is the supported route. |
| #3941 One tree for both runtimes: classic arches to be served by main + reconverse glue, not by resurrecting classic in the branch | 2026 | - | F |  | One tree for both runtimes. Item 2 (hello example) fixed in e498b426a; item 1 superseded by the decision to freeze main as `classic` and make this line the default, which drops the one-tree goal. |
| #3957 GPU D2D: sender blocks on hapiStreamSynchronize in the intra-process path | 2026 | - | K |  | Intra-process device zerocopy blocks the PE: `hapiStreamSynchronize` per buffer on the MEMCPY path (ckrdmadevice.C:1366, 1424). This serialises overdecomposed chares; the fix needs deferred metadata sends from charmxi-generated code. |
| #3960 Device zerocopy (`nocopydevice` / `CkDeviceBuffer`) registers the GPU buffer per message on both ends and never deregisters; long GPU-direct runs die with ENOSPC | 2026 | - | F |  | Device zerocopy registration leak (ENOSPC) fixed by #3961 (b3e5799d0): release on completion, `CkDeviceBufferRegister`, piggybacked acks. The pool (`CkDeviceMalloc`) is present in ckrdmadevice.C. |
| #3963 Device zerocopy examples repack send buffers with nothing guaranteeing the neighbour's get has finished reading them | 2026 | - | K |  | Partly fixed: `jacobi3d` double-buffers (#3980), but `jacobi2d`, `jacobi2d-imbalance`, `verify`, `sdag` and the benchmarks still repack send buffers with no source callback. The manual text flags them as not-a-model. |
| #3964 Host zerocopy path: bundle deregistration acks instead of one NcpyOperationInfo message per buffer | 2026 | - | I |  | Host zerocopy: give dereg acks the device path's id+piggyback scheme; optimization (measured on the device path, 4 us to 0.3 us). |
| #3966 Device zerocopy source callbacks: no way to tell which buffer completed, and no per-send completion | 2026 | - | K |  | Device source callback is an empty message, and a zero-argument `when ghostFree[LEFT]()` never matches the refnum, so the program hangs with no diagnostic. No per-send completion either. API footgun on a new API. |
| #3967 Device zerocopy: user documentation and diagnostics owed | 2026 | - | K |  | Docs were delivered by #3979 (manual.rst ~9360-9640), but the diagnostics are still owed: an informative ENOSPC message, a registration-pressure warning, and a print when same-node traffic falls back to the NIC without `+gpushm`. `hapiGetStream` is still documented rather than deprecated. |
| #3970 Persistent device messaging (CkDevicePersistent) aborts for inter-node transfers | 2026 | - | K |  | `CkDevicePersistent` aborts for inter-node get/put (ckrdmadevice.C:1086/1121, also :153/:186). A stale handle after migration is undetected and can read freed device memory or give silent wrong answers; the proposed pup guard was never added. |
| #3971 src/arch/common/*.sh file directly references compiler, always getting system version- | 2026 | - | K |  | `src/arch/common/cc-*.sh` hardcode compiler names (`cc-clang.sh`: `CMK_CC="clang$CMK_COMPILER_SUFFIX"`), so charmc picks up the PATH's compiler rather than the one CMake was given; still true for reconverse arches. |
| #3972 GPU helper threads inherit a PE's single-core affinity mask under +pemap, costing 22-63% of GPU iteration time | 2026 | - | K |  | GPU runtime helper threads inherit a PE's single-core mask under `+pemap` (-22% to -63% iteration time on Frontier); no fix in reconverse's cpuaffinity (last changes #244/#249 are unrelated). |
| #3973 hapi_memory_daemon: EOF path spins at 1 kHz for the life of the run; EINTR path spins unthrottled | 2026 | - | O |  | `hapi_memory_daemon.cpp` spin. The file is not on the reviewed line (or main); it exists only on `origin/paw-atm26-shrinkexpand`. Matters only if that GPU shrink/expand work merges. |
| #3976 GPU load balancing: improve accuracy of device loads for a mix of kernels; idle-time-based incremental balancing | 2026 | - | I |  | GPU LB load accuracy for mixed kernels (SM-seconds vs chain), idle-time incremental balancing; follow-up design on #3968. |
| #3978 zerocopy_with_qd: intermittent early quiescence in the two-process run on reconverse (macOS CI, 1 of ~10) | 2026 | - | K | * | Intermittent early quiescence in `zerocopy_with_qd` (2 processes, Direct API): QD fires before the RMA completion callbacks, the same window reconverse #222 touched. QD correctness, i.e. silent wrong results for any app ending on QD. |
| #3981 Device zerocopy post API: the regular entry method receives the sender's device pointer, not the posted buffer (RDMA path) | 2026 | - | F |  | Device post API delivered the sender's pointer; fixed by #3999 (08d0e4e83, retargets delivered `CkDeviceBuffer::ptr` to the posted buffers). |
| #3983 CUDA build: host zerocopy acks routed to the device handler (setNcpyOpInfo declared with 24 parameters, defined with 25) | 2026 | - | F |  | `setNcpyOpInfo` arity mismatch fixed by #3984 (359eb90fe; cmirdmautils.h:90 now passes `deviceRdmaOpInfo`). Open residual, tracked only here: 2-proc x 2-PE `d2dtest` still prints "ack for unknown id" (multi-PE device acks resolved on the wrong PE). Needs its own issue before this one closes. |
| #3987 macOS CI: intermittent startup hang in `lcrun -n 2 ./megatest +pe 4` (reconverse-darwin-arm64, 2-process step) | 2026 | - | K |  | `lcrun` startup hang is LCI's file-PMI shared directory (uiuc-hpc/lci#202). Fixed upstream in lci#203 (2026-09-21), but reconverse pins LCI `ca88ce2` (2026-08-06), so the fix is not in; only CI works around it (b26e48b51). Users on laptops hit it after any failed run. Needs an LCI pin bump. |
| #3988 tests/util and tests/ampi do not build on reconverse; make -C tests test cannot pass on this line | 2026 | - | K |  | `make -C tests test` cannot pass on reconverse: tests/Makefile:7,10 still includes `util` (CmiFloat4 missing from reconverse's converse.h) and `ampi` (no ampicc) with no reconverse skip. |
| #3989 Zerocopy post API: posting a buffer smaller than the source is allowed but unreported; larger is unchecked on the host path. Decide the contract. | 2026 | - | K |  | `CkPostBuffer` contract: item 1 done (abort when the posted size exceeds the source, ckrdma.C:987/2432/2536). Still open: the regular EP receives the sender's size after a shorter post; the prefix semantics need a decision; a late second post of the same tag double-delivers and double-frees. |
| #3993 CI: ucx-linux-x86_64_openpmix fails intermittently in ConverseInit with "ucp_ep_create failed: Invalid parameter" (3 times in 3 days) | 2026 | - | O |  | Intermittent `ucp_ep_create` failure in the classic `ucx-linux-x86_64_openpmix` CI job; classic UCX + CI on main. |
| #3994 Revisit the 64-bit object id design: bit allocation, index-to-id mapping, and when an array is compressible | 2026 | - | K | * | 64-bit object id design. Silent correctness hole: past 65,535 ids per PE per collection, the unguarded `idCounter` carries into the home bits and ids collide across PEs (255 in `--ampi-only`). Adaptive refinement reaches this. Guard PR #4016 and redesign #4015/#4017 are open, not merged. |
| #3997 Message forwarding to migrated elements has no hop cap; the send-back-home limit has never been enabled | 2026 | - | K |  | Forwarding to migrated elements has no hop cap: the send-back-home block is still commented out (ckarray.C:1881-1888). Unbounded chase during migration bursts; performance, not correctness. |
| #4008 Classic UCX SMP: exit hangs after "End of program" when sends are still in flight (decided: leave unfixed) | 2026 | Bug,wontfix,UCX,classic | O |  | Classic UCX SMP exit hang with in-flight sends; decided wontfix, `classic` label, layer not on this line. |
| #4018 reconverse: checkpoint restart hangs with >1 process and >1 PE per process (readonly broadcast relay deadlocks in _initDone) | 2026 | - | K | * | Checkpoint restart hangs with more than one process and more than one PE per process on reconverse: `CkRestartMain` calls `_initDone()` only on rank 0 (ckcheckpoint.C:911) and the readonly relay deadlocks. ChaNGa hit it. Fix PR #4019 is open. |

## Counts

| bucket | count |
|---|---|
| K | 31 |
| F | 12 |
| O | 21 |
| A | 1 |
| I | 20 |
| D | 9 |
| total | 94 |

Critical K: 5

## Ranked K items (critical first)

1. **#3762\*** Use-after-free in `CkArray::staticSpringCleaning`, most likely because the Ccd cancel-by-index misses after the list compacts. Any array without stable locations that is destroyed after it has seen one spring-cleaning cycle (one minute) leaves a live callback on a freed `CkArray`. This affects both runtimes and is a one-file fix in conv-conds (reconverse `src/conv-conds.cpp:60-65,115-133,240,254`).
2. **#4018\*** Checkpoint restart hangs with more than one process and more than one PE per process on reconverse (ChaNGa hit it). Fix PR #4019 is open and should merge before the default-branch switch.
3. **#3994\*** Past 65,535 ids per PE, the unguarded per-PE element-id counter silently collides ids across PEs, so dynamic-insertion workloads get wrong delivery. The guard PR #4016 can merge ahead of the redesign.
4. **#3978\*** Intermittent early quiescence in the 2-process Direct-API zerocopy test. QD firing early gives silent wrong results to apps that end on QD.
5. **#3872\*** CkIO `io_read` segfaults on macOS on both runtimes: a broadcast reaches an element with no `CkLocRec` after the reads complete. Reproduced today and absent from the reconverse CI tier.
6. **#3987** `lcrun` bootstrap hang after any failed run; the LCI fix (lci#203) is not in the pinned LCI `ca88ce2`. Bump the pin.
7. **#3970** `CkDevicePersistent`: inter-node transfers abort, and a stale handle after migration can read freed device memory with no diagnostic.
8. **#3966** A zero-argument device source-callback EP never matches its refnum, so the program hangs silently.
9. **#3989** Post API contract: after a shorter post the EP gets the sender's size, and a late second post of the same tag double-delivers and double-frees.
10. **#3501** User entry methods still run after a non-PE0 `CkExit`, against the manual's guarantee.
11. **#3832** At -O0, `[local]`/`[inline]` EPs dereference an uninitialised `curPeEvent` on non-tracing builds, and crash.
12. **#3972** GPU runtime helper threads inherit single-core `+pemap` masks: 22-63% iteration cost at scale on Frontier.
13. **#3957** The intra-process device zerocopy path blocks the PE on `hapiStreamSynchronize` and serialises overdecomposed chares.
14. **#3674** Nodegroup messages starve behind local work. The reconverse scheduler weights the node queue 1:16.
15. **#3988** `make -C tests test` cannot pass on reconverse because `tests/util` and `tests/ampi` have no skip. This blocks a green full-folder site validation.
16. **#3967** Device zerocopy diagnostics are still owed: the ENOSPC message, a registration-pressure warning, and the `+gpushm` fallback print.
17. **#3963** gpudirect examples other than jacobi3d still repack send buffers unsafely.
18. **#3613** Projections tracing and shrink/expand still abort at startup together (init.C:1886).
19. **#3854** `--install-prefix` installs are not relocatable: `conv-mach-opt.sh` sources build-tree paths, which breaks Spack.
20. **#3842** CMake 3.28+ identifies Cray PE compilers as `CrayClang`, which hits the "Unknown compiler" fatal error.
21. **#3845** An unset `NETWORK` crashes CMake, and the default is `netlrts`.
22. **#3846** A direct-CMake shared build gets non-PIC hwloc.
23. **#3971** `cc-*.sh` hardcode compiler names and ignore CMake's CC/CXX.
24. **#3997** Forwarding to migrated elements has no hop cap (performance).
25. **#3934** trace-summary `.sum` and `.sumd` disagree per interval.
26. **#3933** `.sts` `PROJECTIONS_ID` is always empty, and provenance is incomplete.
27. **#3561** MSA `matmul` and `moldyn` examples are broken.
28. **#3862** `ckout << long double` does not compile.
29. **#3803** Dead gerrit URLs are cited throughout the docs.
30. **#3789** Strict-aliasing type pun in `pup_toNetwork.h:117` (the rest of the report is classic code).
31. **#3790** ASan builds fail on charmxi leaks.

## Notes for the caller

- The #3762 mechanism was found by reading the code and has not been tested. Nothing was filed or commented.
- #3860 (a message drop after about 2^31 messages, QD silent) is classed D, but it would be critical if the attached reproducer fails on reconverse. It is worth one run.
- #3983's fix is merged, but the issue is the only record of the multi-PE device "ack for unknown id" residual. Split that residual out before closing it.
- #3973's daemon code exists only on `origin/paw-atm26-shrinkexpand`.
