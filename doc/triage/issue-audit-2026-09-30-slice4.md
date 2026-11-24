# Issue audit, slice 4 (97 issues, #2075-#3460)

Target: `origin/reviewed-with-reconverse` at ebdc49c66 (reconverse submodule b30ad31). Audited 2026-09-30.

Buckets: K keep, F fixed or superseded, O obsolete on the reviewed line, A AMPI/TCharm port list, I idea or wishlist, D needs the reporter or a design call. A star marks a critical K. `(doc)` marks a K that is a documentation gap.

| issue | year | labels | bucket | critical? | reason / evidence |
|---|---|---|---|---|---|
| [#2075](https://github.com/charmplusplus/charm/issues/2075) Improve tuple reduction interface | 2019 | enhancement | I |  | Tuple API ergonomics: `CkReductionMsg::toTuple` still heap-allocates the tuple for the caller to delete[] (ckreduction.C:1601); no defect, a leak-prone interface |
| [#2251](https://github.com/charmplusplus/charm/issues/2251) report both actual host topology and accessible PUs | 2019 | provisioning | I |  | Startup topology banner request. Reconverse has commented out the classic "Running on N hosts" print (conv-topology.cpp:264), so this applies to whatever banner reconverse adds |
| [#2294](https://github.com/charmplusplus/charm/issues/2294) Post-GitHub migration updates | 2019 | cleanup | O |  | 2019 GitHub-migration admin checklist (Gerrit server, old PRs); not code |
| [#2295](https://github.com/charmplusplus/charm/issues/2295) `CcdCallFnAfterOnPE` doesn't execute on requested PE on most builds | 2019 | - | F |  | Reconverse `CcdCallFnAfterOnPE` queues onto the target PE (conv-conds.cpp:269), and the reconverse test tests/conds/conds.cpp asserts it ran on the requested PE |
| [#2325](https://github.com/charmplusplus/charm/issues/2325) Add support for plotting performance data from nightly autobuild's benchmark runs | 2019 | support, Build & test automation | O |  | Plotting for the old nightly autobuild; CI is GitHub Actions on the reviewed line |
| [#2339](https://github.com/charmplusplus/charm/issues/2339) NAMD pami build on summit does not print stack trace on segfault or abort | 2019 | - | O |  | pami/pamilrts CmiAbort without a stack trace; the PAMI layer is not built |
| [#2347](https://github.com/charmplusplus/charm/issues/2347) Deleting the migration message in a mainchare migration ctor after restart causes error. | 2019 | documentation | F |  | Documented by #2348 (2da23830d): users must not free the migration-constructor message |
| [#2365](https://github.com/charmplusplus/charm/issues/2365) Change default semantic to not support anytime migration | 2019 | - | I |  | Proposal to make anytime migration opt-in; default is still `_isAnytimeMigration = true` (init.C:383) |
| [#2366](https://github.com/charmplusplus/charm/issues/2366) Change default semantic to not support anytime insertion | 2019 | - | I |  | Proposal to make anytime insertion opt-in; design choice, still applicable |
| [#2379](https://github.com/charmplusplus/charm/issues/2379) Make [nokeep] semantic the default for broadcasts (and maybe more) | 2019 | - | I |  | Proposal to make message entries [nokeep] by default. Marshalled entries already are; the 2026-09 audit confirmed message entries still copy once per receiver |
| [#2422](https://github.com/charmplusplus/charm/issues/2422) Reduce memory usage for default spanning tree scheme | 2019 | - | K |  | `ST_RecursivePartition_getTreeInfo` (src/util/spanningTree.C:622) still caches one tree per distinct root in a process-wide map that is never freed; ZC and group broadcasts from many roots grow it without bound |
| [#2447](https://github.com/charmplusplus/charm/issues/2447) Record/replay failure to read replay file in megampi  | 2019 | - | A |  | Record/replay "syntax error reading replay file" in AMPI megampi (netlrts-smp); recheck when AMPI is ported (record/replay itself works on reconverse) |
| [#2477](https://github.com/charmplusplus/charm/issues/2477) UCX: applications do not work without charmrun/mpirun | 2019 | UCX | O |  | UCX layer fails without charmrun; UCX is not built |
| [#2487](https://github.com/charmplusplus/charm/issues/2487) Handle corner cases correctly when using ZC Bcast Post API for chare arrays | 2019 | Zerocopy | K |  | ZC Bcast Post API for arrays when a node has 0 elements is still unhandled: `//TODO: Equeue the basic message if there are no elements` (ckrdma.C:1697); risk of a hang or a lost message |
| [#2491](https://github.com/charmplusplus/charm/issues/2491) Automatically check for and deal with old build files during build | 2019 | - | K |  | Stale *.decl.h/*.def.h files in src/ still shadow generated ones: CMake compiles ck-libs sources in place and quoted includes search the source directory first. No build-time check exists |
| [#2511](https://github.com/charmplusplus/charm/issues/2511) Demand creation can hang in the presence of non-demand creation messages. | 2019 | - | F |  | Demand creation now has its own request path (`bufferForCreation` -> `requestDemandCreation`, ckarray.C:2100-2150, from #3266/#3313), separate from location requests |
| [#2512](https://github.com/charmplusplus/charm/issues/2512) Demand creation crashes when setting bounds on the array | 2019 | - | F |  | Same refactor: `requestDemandCreation` approves creation when `!lookupID(...)` or `whichPe(id)==-1`, so bounded arrays (index to ID is known) are handled |
| [#2549](https://github.com/charmplusplus/charm/issues/2549) CMK_LBDB_ON usages should be cleaned up | 2019 | cleanup, load balancing | I |  | Cleanup only: 236 `CMK_LBDB_ON` uses remain in src/ |
| [#2573](https://github.com/charmplusplus/charm/issues/2573) syncft: incompatibilities with SMP mode | 2019 | smp, fault tolerance | O |  | syncft (message-logging/in-memory fault tolerance) with SMP; fault tolerance is a tombstone |
| [#2574](https://github.com/charmplusplus/charm/issues/2574) syncft: recovery from node 0 fault fails in some circumstances | 2019 | fault tolerance | O |  | syncft recovery from a node-0 fault; tombstoned feature |
| [#2575](https://github.com/charmplusplus/charm/issues/2575) syncft: recovery fails on verbs layer | 2019 | fault tolerance, Verbs | O |  | syncft on verbs; verbs and fault tolerance are both gone |
| [#2576](https://github.com/charmplusplus/charm/issues/2576) syncft: recovery from multiple simultaneous process failures is unsupported | 2019 | fault tolerance | O |  | syncft with multiple simultaneous failures; tombstoned feature |
| [#2577](https://github.com/charmplusplus/charm/issues/2577) syncft: MPI spare processors do not fully initialize before waiting, breaking isomalloc_sync | 2019 | AMPI, fault tolerance, MPI layer | O |  | MPI-layer spare processors (+wp) skip isomalloc_sync; the MPI layer and syncft are both gone (it carries the AMPI label, but the defect is in the MPI layer) |
| [#2578](https://github.com/charmplusplus/charm/issues/2578) Shrink-Expand: SMP mode is unsupported | 2019 | smp, shrink-expand | O |  | Shrink/expand on netlrts-smp (CCS plus charmrun restart); the reviewed line does shrink/expand through checkpoint, and netlrts is not built |
| [#2582](https://github.com/charmplusplus/charm/issues/2582) multicore support for Windows machines with more than 64 processors | 2019 | Windows | O |  | Windows multicore with more than 64 PUs; reconverse has no Windows port |
| [#2593](https://github.com/charmplusplus/charm/issues/2593) Add support to test the big 3 charm++ applications (NAMD, ChaNGa, Openatom) through autobuild | 2019 | support, Build & test automation | I |  | Test NAMD/ChaNGa/OpenAtom in CI; matches the planned non-blocking apps-ci.yaml |
| [#2613](https://github.com/charmplusplus/charm/issues/2613) Checkpoint/restart hangs in SMP mode | 2019 | checkpoint/restart | D |  | chkpt restart with 2 procs x 2 PEs hung on pamilrts (Lassen/Summit); tests/charm++/chkpt is not in reconverse CI. Needs a multi-process restart test with lcrun on reconverse |
| [#2629](https://github.com/charmplusplus/charm/issues/2629) DistributedLB crashes execution on LeanMD mini-app | 2019 | load balancing | F |  | DistributedLB self-migration: #2784 (5adc51c36) stops underloaded PEs from transferring load, and #2639 (95abe338a) aborts on self-moves in DistBaseLB |
| [#2636](https://github.com/charmplusplus/charm/issues/2636) ChaNGa crashes/hangs with UCX machine layer (in SMP mode) on Frontera | 2019 | Bug, UCX | O |  | ChaNGa crashes on UCX SMP (ibv_reg_mr failures); UCX is not built |
| [#2643](https://github.com/charmplusplus/charm/issues/2643) pupping a proxy fails when restarting from a checkpoint | 2019 | - | K | ★ | `CkLocalNodeBranch` still spins `CsdScheduler(0)` until the nodegroup exists (ck.C:536-546); unpacking a delegated proxy during restart re-enters the scheduler and delivers entry methods to half-unpacked elements (ChaNGa crash) |
| [#2677](https://github.com/charmplusplus/charm/issues/2677) syncft: only Chare Arrays get recreated during restart on non-faulted nodes | 2020 | fault tolerance | O |  | syncft: only arrays recreated on surviving nodes; tombstoned feature |
| [#2678](https://github.com/charmplusplus/charm/issues/2678) Migratable mainchares and singleton chares don't have ckJustMigrated | 2020 | migration | K |  | `ckJustMigrated` is declared on IrrGroup (charm++.h:354) and on array elements but not on `class Chare`, so a migratable mainchare or singleton cannot override it |
| [#2706](https://github.com/charmplusplus/charm/issues/2706) megacon ringsimple exposes heap-use-after-free in netlrts-smp | 2020 | - | O |  | ASan heap-use-after-free in netlrts machine-eth.C; netlrts is not built |
| [#2718](https://github.com/charmplusplus/charm/issues/2718) Evaluate message queue/priority defaults  | 2020 | - | I |  | Evaluate queue and priority defaults; design study |
| [#2722](https://github.com/charmplusplus/charm/issues/2722) Document or remove [iget] entry method attribute | 2020 | cleanup, documentation | K |  | [iget] is still in charmxi (xi-Entry.C:460) and only listed as a reserved word in the manual (manual.rst:13011); document it or remove it (doc) |
| [#2726](https://github.com/charmplusplus/charm/issues/2726) CkLocMgr Refactor/NodeAware location cache | 2020 | - | I |  | Node-aware location cache design discussion; overlaps Kale's queued location-manager and process-level cache work |
| [#2733](https://github.com/charmplusplus/charm/issues/2733) no error on bogus pe map | 2020 | - | K |  | Reconverse `search_pemap` (cpuaffinity.cpp:66) never checks the `sscanf(str,"%d",&start)` result, so `+pemap foo` reads an uninitialized `start` with no error |
| [#2743](https://github.com/charmplusplus/charm/issues/2743) Add CkCallback type using function registration scheme | 2020 | feature | I |  | Register callback functions by index instead of raw pointers in CkCallback (`callCFn` still stores `CkCallbackFn fn`); feature |
| [#2747](https://github.com/charmplusplus/charm/issues/2747) Exhaustively document most runtime flags  | 2020 | - | K |  | Manual still does not list most runtime flags (no +CmiSleepOnIdle/+CmiSpinOnIdle); reconverse flags are also undocumented (doc) |
| [#2758](https://github.com/charmplusplus/charm/issues/2758) Re-think .ci/module structure | 2020 | charmxi | I |  | Rethink .ci module structure; design idea |
| [#2760](https://github.com/charmplusplus/charm/issues/2760) OFI layer fails on Frontera with an assertion failure in LrtsInit | 2020 | OFI | O |  | OFI layer PMI failure on Frontera; OFI is not built |
| [#2765](https://github.com/charmplusplus/charm/issues/2765) Evaluate Migratability defaults | 2020 | - | I |  | Make arrays static by default; design idea |
| [#2777](https://github.com/charmplusplus/charm/issues/2777) Make ZC Pup complete inline | 2020 | - | A |  | ZC pup inline completion; motivated by AMPI Isomalloc (#2707) |
| [#2779](https://github.com/charmplusplus/charm/issues/2779) Extricate LocalBarrier from load balancing code | 2020 | cleanup, load balancing | F |  | Done by #2955 (eee6679c8): the AtSync barrier moved out of LBManager into CkSyncBarrier |
| [#2796](https://github.com/charmplusplus/charm/issues/2796) Documentation: Setting the bounds on array index spaces is not in the manual | 2020 | documentation | K |  | Manual still does not document array bounds (no `setBounds` in manual.rst) (doc) |
| [#2797](https://github.com/charmplusplus/charm/issues/2797) Array index compression is not currently an option for user defined index types | 2020 | - | I |  | Allow a user-supplied index compressor; only `FixedArrayIndexCompressor::make(bounds)` exists (cklocation.C:2476). Relevant to AMR oct-tree indices |
| [#2807](https://github.com/charmplusplus/charm/issues/2807) Add documentation to enumerate the different RTS array maps | 2020 | documentation | K |  | Manual documents only BlockMap/RRMap by example; RTS array maps are not listed (doc) |
| [#2811](https://github.com/charmplusplus/charm/issues/2811) Modernize/simplify CkArrayIndex | 2020 | - | I |  | Modernize or template CkArrayIndex; cleanup idea |
| [#2818](https://github.com/charmplusplus/charm/issues/2818) Metabalancer: Add support for DistributedLB | 2020 | - | I |  | MetaBalancer support for DistributedLB; feature |
| [#2822](https://github.com/charmplusplus/charm/issues/2822) Cleanup compare functors used in TreeLB strategies | 2020 | cleanup, load balancing | I |  | Parameterize TreeLB comparison functors; cleanup |
| [#2824](https://github.com/charmplusplus/charm/issues/2824) Occasional hang in converse kNeighbors and randomTtl with pamilrts-smp builds on Lassen | 2020 | - | O |  | pamilrts-smp hang in kNeighbors/randomTtl; PAMI is not built |
| [#2828](https://github.com/charmplusplus/charm/issues/2828) Incorrect values returned by CmiRankOf and CmiNodeOf on the comm thread  | 2020 | - | O |  | `CmiRankOf`/`CmiNodeOf` wrong on the comm thread; reconverse has no comm thread |
| [#2831](https://github.com/charmplusplus/charm/issues/2831) idx2str does not work as expected | 2020 | Bug | K |  | `idx2str` still returns one static buffer (charm++.h:1312), so two calls in one printf print the last index twice; not thread-safe in SMP |
| [#2839](https://github.com/charmplusplus/charm/issues/2839) Fix CMake errors/unsupported configs | 2020 | CMake | K |  | CMake gap tracker, still relevant on a CMake-only line. Open items: the `$(MAKE)` in cmake/hwloc.cmake:48 still breaks Ninja, OpenMP (#3396), ParFUM, and tsan/nolb/ooc options |
| [#2849](https://github.com/charmplusplus/charm/issues/2849) NAMD occasionally crashes on Frontera with mpi-smp builds | 2020 | - | O |  | NAMD mpi-smp crash from an uninitialized `zcMsgType` in the classic MPI header; reconverse sets `zcMsgType = CMK_REG_NO_ZC_MSG` at allocation (convcore.cpp:1042) |
| [#2897](https://github.com/charmplusplus/charm/issues/2897) require ++ppn with ++mpiexec-no-n for smp charmrun | 2020 | - | O |  | charmrun ++mpiexec-no-n with +ppn; charmrun is not used |
| [#2923](https://github.com/charmplusplus/charm/issues/2923) Make AMPI-only the default AMPI version | 2020 | AMPI | A |  | Build an AMPI-only library in CMake |
| [#2926](https://github.com/charmplusplus/charm/issues/2926) Fully document installation options | 2020 | documentation | K |  | `./build --help` runs `buildold --help` (buildcmake:51-53), which lists classic layers and options and never mentions reconverse; install options for the reviewed line are undocumented (doc) |
| [#2932](https://github.com/charmplusplus/charm/issues/2932) libgfortran heap-allocates format strings as hash table keys at runtime | 2020 | AMPI, migration, Fortran | A |  | libgfortran format-string keys on the rank's Isomalloc heap break migration |
| [#2947](https://github.com/charmplusplus/charm/issues/2947) charmrun examples should use ++n instead of +p | 2020 | - | O |  | Manual charmrun examples should use ++n; charmrun is not used |
| [#2964](https://github.com/charmplusplus/charm/issues/2964) ++debug reporting single PE for provisioning | 2020 | Bug, charmrun, ease of use | O |  | charmrun ++debug provisioning message; charmrun is not used |
| [#2974](https://github.com/charmplusplus/charm/issues/2974) Segfault in global variable destructors with -memory gnu | 2020 | AMPI | A |  | AMPI benchmarks crash at exit with -memory gnu |
| [#2988](https://github.com/charmplusplus/charm/issues/2988) Make build script choose the best possible arch based on the machine without requiring user input  | 2020 | - | I |  | Auto-select the build target; less needed with a single machine layer, but a triplet is still required |
| [#2991](https://github.com/charmplusplus/charm/issues/2991) Add tests for AMR library  | 2020 | good first issue, support | D |  | libs/ck-libs/amr is still `EXCLUDE_FROM_ALL` and untested (src/libs/ck-libs/CMakeLists.txt:199); needs a decision to keep and test AMR or delete it |
| [#3071](https://github.com/charmplusplus/charm/issues/3071) Improve Group List Send | 2020 | enhancement, machine layers, smp, Converse | F |  | Superseded: reconverse's list send now fans out once per destination process and copies once per process (reconverse #238, c1070b5); the classic Converse/LRTS races do not apply |
| [#3119](https://github.com/charmplusplus/charm/issues/3119) Document pup_buffer API | 2020 | documentation | K |  | pup_buffer API is still undocumented (no mention in manual.rst) (doc) |
| [#3143](https://github.com/charmplusplus/charm/issues/3143) Pup Buffer Test failure on ofi-smp | 2020 | - | O |  | pup_buffer test failure on ofi-smp autobuild; OFI is not built (pup_buffer passes in reconverse CI) |
| [#3158](https://github.com/charmplusplus/charm/issues/3158) SDAG: Refnum matching requires taking msg ptr as parameter | 2020 | Bug, charmxi | K | ★ | Confirmed with the local charmxi: for `when foo[3](void)`, the generated `_call_foo_void` drops the message and builds a closure with no refnum, so a refnum-matched void entry never matches and hangs (message-typed entries work) |
| [#3178](https://github.com/charmplusplus/charm/issues/3178) Feature Request: Adding continuations to charm++ futures | 2020 | - | I |  | `future.then()` continuations; feature |
| [#3180](https://github.com/charmplusplus/charm/issues/3180) User-defined types in array indices | 2020 | feature | I |  | First-class user-defined array index types (proposed `as_index`); feature, not implemented |
| [#3181](https://github.com/charmplusplus/charm/issues/3181) Unable to build with shared libraries and gcc on MacOS | 2020 | MacOS | D |  | macOS gcc --build-shared: undefined Ccd* symbols in libck. On the reviewed line libck links only CUPTI (src/ck-core/CMakeLists.txt:150), not reconverse; untested whether reconverse-darwin --build-shared links |
| [#3210](https://github.com/charmplusplus/charm/issues/3210) Expose charmc's "moduleinit" functionality | 2020 | enhancement, charmc, CMake | I |  | Expose charmc's moduleinit generation to CMake consumers; CharmConfig.cmake has nothing for it |
| [#3212](https://github.com/charmplusplus/charm/issues/3212) LrtsNodeLock has inconsistent behavior when `CMK_SHARED_VARS_UNAVAILABLE` is defined | 2020 | - | O |  | `LrtsNodeLock` in classic machine-common-core.C; not built |
| [#3235](https://github.com/charmplusplus/charm/issues/3235) Bug report: missing vtable with PUPable_decl | 2021 | - | I |  | Not a bug: `PUPable_decl` needs `PUPable_def` in one .C file to emit `get_PUP_ID` (the vtable anchor). The manual could say so explicitly |
| [#3244](https://github.com/charmplusplus/charm/issues/3244) Improve Metabalancer Documentation | 2021 | - | K |  | MetaBalancer model format and training are undocumented (doc) |
| [#3252](https://github.com/charmplusplus/charm/issues/3252) Invalid affinity mapping not causing error | 2021 | Bug, smp | O |  | +commap to an invalid core accepted; reconverse has no comm thread or +commap (cpuaffinity.cpp:601) |
| [#3253](https://github.com/charmplusplus/charm/issues/3253) Warn when +commap causes oversubscription | 2021 | smp, ease of use | O |  | Oversubscription warning for +commap; no comm thread on reconverse |
| [#3260](https://github.com/charmplusplus/charm/issues/3260) cmake: will not rebuild certain targets if only header file(s) changed | 2021 | CMake | D |  | Header copies in include/ are made by configure_file COPYONLY. A touch-only change does not refresh the copy, but a content edit should; needs a quick check with a real edit on the reviewed line |
| [#3270](https://github.com/charmplusplus/charm/issues/3270) Support deleting objects while in LB | 2021 | enhancement, load balancing | I |  | Allow deleting objects during LB (`LBDatabase::Migrate` still aborts on stale handles); feature |
| [#3309](https://github.com/charmplusplus/charm/issues/3309) AMPI: Fix intercommunicator creation procedure | 2021 | AMPI | A |  | AMPI intercommunicator creation procedure |
| [#3311](https://github.com/charmplusplus/charm/issues/3311) cmake ignores additional include/library directories while configuring project | 2021 | CMake | O |  | CMake --basedir not honored when detecting OFI; OFI layer is not built (reconverse finds libfabric through its own CMake) |
| [#3351](https://github.com/charmplusplus/charm/issues/3351) Section reductions require CkMulticast, can't update cookie when dispatching reduction from an array element | 2021 | - | K |  | `CkGetSectionInfo` still ignores cookies from simple sends (`if (m->gpe() != -1)`, ckmulticast.C:1145), so an element cannot update its section cookie without a CkMulticast message |
| [#3365](https://github.com/charmplusplus/charm/issues/3365) No Error When Sending Messages of Unregistered Types | 2021 | - | K |  | `envelope::setMsgIdx` (envelope.h:239) still does no validity check, and `CMessage_CkMessage::__idx=-1`; unregistered message types fail later and obscurely |
| [#3366](https://github.com/charmplusplus/charm/issues/3366) Anytime migration happen when flag is disabled | 2021 | migration | K |  | With +noAnytimeMigration, `ckEmigrate` only prints a warning and migrates anyway (ckarray.h:368-374), even though arrays with static insertion and no anytime migration set `stableLocations` (ckarray.C:876) |
| [#3380](https://github.com/charmplusplus/charm/issues/3380) Normalize locations of external libraries | 2021 | cleanup | I |  | Collect external libraries under contrib/; cleanup |
| [#3396](https://github.com/charmplusplus/charm/issues/3396) OpenMP Library Not Being Built With CMake | 2021 | Bug, openmp integration, CMake | K |  | The CMake `OMP` option only calls find_package; src/libs/conv-libs/openmp_llvm is never built, so omp-smp is unavailable on a CMake-only line |
| [#3399](https://github.com/charmplusplus/charm/issues/3399) Use Linux FSGSBASE support for TLSglobals privatization | 2021 | AMPI | A |  | FSGSBASE for TLSglobals privatization |
| [#3401](https://github.com/charmplusplus/charm/issues/3401) ScotchLB occasionally FPEs | 2021 | Bug, load balancing | K |  | ScotchLB (ScotchLB.C:106-120) still computes `256.0/maxLoad` and `1024.0/maxBytes` with no zero guard and can pass 0 or NaN weights to SCOTCH_graphPart; the FPE is unaddressed |
| [#3408](https://github.com/charmplusplus/charm/issues/3408) Duplication in Cth Thread Tracing  | 2021 | Bug, cleanup, tracing | F |  | Superseded on reconverse: `CthTraceResume` is empty (threads.cpp:476) and tracing is only through listeners and traceAwaken, so the duplicate blocks are gone. Check raw-Cth tracing for NAMD separately |
| [#3413](https://github.com/charmplusplus/charm/issues/3413) CkArray: Better synchronization for begin/doneInserting calls | 2021 | Bug, enhancement | K |  | `remoteDoneInserting` (ckarray.C:1112) still flips a flag with no insertion count or completion detection, so reduction and AtSync counts can be wrong with dynamic insertion |
| [#3414](https://github.com/charmplusplus/charm/issues/3414) CkArray: Better defaults for anytime migration | 2021 | cleanup | I |  | Better anytime-migration defaults; #3623 was a first step; design |
| [#3416](https://github.com/charmplusplus/charm/issues/3416) CkArray: Update and improve dynamic insertion behavior | 2021 | enhancement, cleanup | K |  | Umbrella for dynamic insertion; #3413 (open, K) and #3414 remain |
| [#3422](https://github.com/charmplusplus/charm/issues/3422) Use expedited by default | 2021 | - | I |  | Make [expedited] the default; design and performance choice |
| [#3429](https://github.com/charmplusplus/charm/issues/3429) Negative overhead entry with summary tracemode in details | 2021 | - | K |  | trace-summary still charges pack time to both the enclosing EP and `dummy_pack_ep` (trace-summary.C:882-887), giving negative overhead in +sumDetail |
| [#3447](https://github.com/charmplusplus/charm/issues/3447) Summary/Sum Detail Communication Logs | 2021 | - | I |  | Communication logs in summary/sumDetail; partly covered by open PR #3937 (targets main, not the reviewed line) |
| [#3459](https://github.com/charmplusplus/charm/issues/3459) Deprecate User-managed Section Cookies | 2021 | - | I |  | Section manager so users need not manage section cookies; design |
| [#3460](https://github.com/charmplusplus/charm/issues/3460) Sections Rework | 2021 | - | I |  | Sections rework requirements (cross-array sections, node-aware); design |

## Counts

| bucket | count |
|---|---|
| K | 24 |
| F | 8 |
| O | 26 |
| A | 7 |
| I | 28 |
| D | 4 |
| total | 97 |

## Ranked K items (24)

Critical:

1. ★ **#3158**: an SDAG `when foo[N](void)` never matches, so the program hangs. Checked against code generated by the local charmxi: `_call_foo_void` drops the incoming message, and the closure is built with no refnum. The workaround is to declare the entry with a message parameter.
2. ★ **#2643**: restarting from a checkpoint can crash. `CkLocalNodeBranch` calls `CsdScheduler(0)` while waiting for a nodegroup (ck.C:542-545). Unpacking a delegated proxy during restart therefore runs entry methods on elements that are only partly unpacked (seen in ChaNGa).

Other K, by impact:

3. **#3413**: `doneInserting` does no synchronization (ckarray.C:1112). With dynamic insertion, reduction and AtSync element counts can be wrong. Matters for AMR and other dynamic-insertion codes.
4. **#3416**: umbrella for dynamic insertion. #3413 and #3414 are still open.
5. **#2487**: in the ZC broadcast post API for arrays, a node with no local elements is left as a TODO (ckrdma.C:1697). Possible hang or lost message.
6. **#3366**: `+noAnytimeMigration` only prints a warning and the migration still happens, even though the array has assumed `stableLocations`.
7. **#3401**: ScotchLB divides by `maxLoad` and `maxBytes` without checking for zero. Zero or NaN weights reach SCOTCH_graphPart, which is the likely cause of the reported FPE.
8. **#2422**: the per-root spanning-tree cache (`spanningTree.C:622`) is never freed and grows with every distinct broadcast root.
9. **#3351**: section reductions cannot refresh their cookie outside CkMulticast (ckmulticast.C:1145).
10. **#3396**: on the CMake-only line, the OpenMP runtime integration (openmp_llvm) is never built.
11. **#2839**: tracker of open CMake gaps. The Ninja `$(MAKE)` problem in hwloc.cmake is still there, plus OpenMP and the tsan/nolb/ooc options.
12. **#3365**: sending an unregistered message type is not caught (`setMsgIdx` has no check).
13. **#3429**: summary tracing counts pack time twice, which produces negative overhead.
14. **#2733**: `+pemap foo` reads an uninitialized value in reconverse `search_pemap` and gives no error.
15. **#2491**: stale generated .decl.h/.def.h files in src/ override the freshly generated ones.
16. **#2678**: `ckJustMigrated` is missing on `Chare`, so mainchares and singletons cannot override it.
17. **#2831**: `idx2str` uses a single static buffer, so two calls in one printf print the same index.
18. **#2926** (doc): `./build --help` prints the classic `buildold` help, which lists classic layers and never mentions reconverse.
19. **#2747** (doc): most runtime flags are not documented; reconverse flags need adding.
20. **#2796** (doc): array bounds (`setBounds`) are not documented.
21. **#2807** (doc): the RTS array maps are not listed.
22. **#3119** (doc): the pup_buffer API is not documented.
23. **#2722** (doc): the [iget] attribute needs documenting or removing.
24. **#3244** (doc): the MetaBalancer model format and training are undocumented.

## Notes

- Demand creation (#2511, #2512) looks fixed by the #3266/#3313 location-manager refactor. The code shows a separate `requestDemandCreation` path. It was not rerun.
- #2613 (restart hang in SMP) is D rather than K because tests/charm++/chkpt is not in reconverse CI. A 2-process lcrun restart test would settle it.
- #3408: the duplicate tracing is gone on reconverse. Because reconverse threads never call `traceResume`/`traceSuspend` directly, raw Cth threads without listeners (as NAMD uses) may now go untraced. This is a separate question.
