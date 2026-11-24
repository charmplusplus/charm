# Issue audit, slice 3 (charmplusplus/charm, #1542-#2073)

Checked against `origin/reviewed-with-reconverse` at ebdc49c66 (2026-09-28), with contrib/reconverse at b30ad31. Buckets: K keep, F fixed or superseded, O obsolete on the reviewed line, A AMPI/TCharm port list, I optional idea/wishlist, D needs the reporter or a design call. A star marks a critical K.

| issue | year | labels | title | bucket | critical? | reason / evidence |
|---|---|---|---|---|---|---|
| #1542 | 2017 | Bug | CkArrayCreated callback should be part of CkArrayOptions | F |  | ckarrayoptions.h:275 has `setInitCallback()`; ckarray.C:874,1086 uses `opts.getInitCallback()`. The request is implemented; the old ckNew(cb, opts) overload remains alongside it. |
| #1545 | 2017 | Bug | Serialize std::vector with Custom Allocator | K |  | pup_stl.h:73,341 still defines `operator\|` only for `std::vector<T>` (default allocator), so a vector with a custom allocator does not compile. Stateless-allocator fix is 2 lines. |
| #1548 | 2017 | support | Reassess whether the primary scheduler thread should support CthSuspend | D |  | Design question: should the main scheduler thread be CthSuspend-able? Reconverse threads.cpp:356 aborts on a non-suspendable thread. Needs a design call on reconverse. |
| #1549 | 2017 | feature,tracing | record priorities in traces and use in projections | I |  | Projections does not record message priority (no priority field in trace-projections.C/.h). Nice-to-have for perf debugging. |
| #1550 | 2017 | cleanup,Build & test automation | Missing 'make test' for examples/charm++/state_space_searchengine/ | I |  | examples/charm++/state_space_searchengine has no `test` target, and its Makefile runs `cd searchengineLib`, a directory that does not exist (the lib is in src/libs/ck-libs). Clean up or remove the example. |
| #1562 | 2017 | feature | Enable message allocation, construction, packing, etc, without generated .ci file code | I |  | API evolution: allocate/construct/pack messages without .ci-generated code; also drop `-fno-lifetime-dse`. Long-term design, no defect. |
| #1582 | 2017 | feature,AMPI | DDT support for direct copy from a noncontiguous type to another noncontiguous type | A |  | AMPI DDT: copy directly from one noncontiguous type to another ("transpacking"). |
| #1586 | 2017 | support,openmp integration | OpenMP integration support for external library written in and compiled by OpenMP on Charm++ | D |  | The OpenMP integration (src/libs/conv-libs/openmp_llvm) is not added to the CMake build on the reviewed line (conv-libs/CMakeLists.txt builds only packlib/cms). Decide whether it is kept before this request can apply. |
| #1608 | 2017 | feature | Support send on IDLE in TRAM | I |  | TRAM send-on-idle. Feature request; NDMeshStreamer still ships. |
| #1612 | 2017 | feature | Support for using rdma to return data from a [sync] entry method | I |  | Return data from [sync] entry methods over RDMA to avoid a message plus PUP. Feature request. |
| #1622 | 2017 | feature,AMPI | Optimize AMPI RMA routines for shared memory | A |  | AMPI RMA: node-local shared-memory optimization. |
| #1629 | 2017 | Bug,machine layers | Valgrind shows invalid writes at GNI_SmsgSendWTag calls for charm programs built on gni-craxyc  | O |  | Valgrind invalid writes inside the GNI library (gni-crayxc machine layer). |
| #1634 | 2017 | Bug,AMPI | HDF5 issues in AMPI | A |  | HDF5 on AMPI: migration with open files, spurious crashes at termination. |
| #1635 | 2017 | Bug | Cleanup and Activate the Charm++ Topology Examples | I |  | examples/charm++/topology (jacobi3d, matmul3d) still hit an FPE in ComputeMap and are not in the examples Makefile. Fix is confined to the examples. |
| #1636 | 2017 | Bug | Update the machine layer template configuration in src/arch/template | O |  | src/arch/template is a template for classic LRTS machine layers. Not used on the reviewed line. |
| #1640 | 2017 | Bug,AMPI | Segfault during migration for AMPI in SMP mode with "-tracemode projections" | A |  | AMPI SMP + tracemode projections segfaults in isomalloc pup during migration (inline local sends). |
| #1666 | 2017 | good first issue,feature,openmp integration | Standalone OpenMP implementation based on Converse User-level threads | D |  | Standalone OpenMP on Converse. The task queue now lives in reconverse (conv-taskq.cpp), but openmp_llvm is not in the CMake build. Needs the same keep-or-drop call as #1586. |
| #1671 | 2017 | Bug,machine layers | Verbs memory pool may leak pinned memory when message is deleted on a PE different from the one to which it was delivered | O |  | Pinned-memory leak in the verbs machine-ibverbs.c memory pool. |
| #1677 | 2017 | feature,machine layers | improved topology-aware partitioner | I |  | Better topology-aware +partitions partitioner. Reconverse CmiCreatePartitions (convcore.cpp:1880) does only simple partitioning, and the classic recursive bisection in machine-common-core is not built. Research idea. |
| #1681 | 2017 | support,documentation | Document Exception Handling in Charm++ | F |  | std::set_terminate is installed in init.C:1329. The noexcept part is tracked as #1729. A manual section on exceptions is still missing (small doc item). |
| #1686 | 2017 | Bug | Use a namespace for Charm++ code | I |  | Put Charm++ in a namespace and stop using reserved identifiers (`__register`, `_registerX`). Only a few headers use `namespace ck`. API-breaking design item. |
| #1693 | 2017 | feature | Print template parameters when missing explicit instantiation | I |  | Missing-instantiation abort (register.C:50-51) prints the EP name and address only, not the chare class or template arguments. Usability improvement. |
| #1695 | 2017 | feature | Allow "type aliases" for explicit instantiations of member function templates | I |  | Allow type aliases in .ci explicit instantiations of template entry methods. charmxi feature. |
| #1697 | 2017 | feature | Distinguishing local object calls from entry method calls | I |  | Let an entry method know whether it was called locally or through a proxy. Feature request. |
| #1700 | 2017 | Bug | Overloaded reduction targets result in compilation error | K |  | xi-Entry.C:1497 emits `redn_wrapper_<name>(CkReductionMsg*)` with no overload suffix, so two overloaded [reductiontarget] methods generate a redeclaration compile error. |
| #1707 | 2017 | Bug | Nocopy Entry method API ack handling crashes on pamilrts-bluegeneq-async-smp | O |  | Zerocopy ack crash on pamilrts-bluegeneq (PAMI layer). |
| #1710 | 2017 | Bug | syncft tests: warning and crash on init_checkpt | O |  | syncft (in-memory checkpoint) killFile test crash on netlrts via charmrun process restart. Fault-tolerance and charmrun path. |
| #1711 | 2017 | Bug | syncft tests: unclear failure | O |  | Same syncft/charmrun restart test failure as #1710. |
| #1713 | 2017 | feature,AMPI | DDT support for getting the addresses of contiguous parts of non-contiguous datatypes | A |  | AMPI DDT: iterate over the contiguous pieces for zerocopy rget/rput. |
| #1725 | 2017 | support,Build & test automation | Improve pup_stl testing | I |  | Broaden pup_stl testing. examples/charm++/PUP/STLPUP still covers only std::vector. |
| #1729 | 2017 | feature | Mark the entire RTS noexcept | I |  | Mark the RTS noexcept, allowing -fno-exceptions. Optional optimization. |
| #1736 | 2017 | Bug | pami-linux-ppc64le-async-smp programs crash on Summitdev with an assertion failure | O |  | pami-linux-ppc64le async-smp assertion on Summitdev. |
| #1737 | 2017 | Bug | tests/charm++/pingpong and examples/charm++/zerocopy/pingpong fail when run on 2 processes with an assertion failure on pamilrts-linux-x86_64 | O |  | pamilrts-linux-x86_64 pingpong CMA assertion. |
| #1740 | 2017 | Bug | Failure at LrtsInit with OFI build with gni provider on Edison | O |  | OFI layer with the gni provider fails in LrtsInit on Edison. |
| #1759 | 2017 | Bug | cmiArgDebugFlag Used Before Initialization | O |  | cmiArgDebugFlag used before init in the classic conv-core/machine-common-core abort path. Reconverse CmiAbort (convcore.cpp:1133) does not read it. |
| #1762 | 2017 | cleanup | Resolve circular dependence in charmc's linking of memory/thread libraries | A |  | charmc links memory/thread libs twice because -tlsglobals (AMPI privatization) broke when the duplicate was removed. Revisit with the AMPI port. |
| #1763 | 2017 | feature,AMPI | Generate libmpi.{a,so} for AMPI configure scripts | A |  | Produce libmpi.{a,so} / a single libampi.so for configure scripts. |
| #1772 | 2018 | Bug | Programs built on pami-linux-ppc64le-async-smp fail a low level assertion at runtime.  | O |  | pami-linux-ppc64le async-smp assertion (PAMI). |
| #1773 | 2018 | Bug | Zerocopy examples fail on pami-linux-ppc64le-smp due to a low level assertion failure | O |  | Zerocopy examples fail on pami-linux-ppc64le-smp (PAMI CMA assertion). |
| #1775 | 2018 | Bug | chkpt test hangs for pami-linux-ppc64le-smp & pamilrts-linux-ppc64le-smp build | O |  | chkpt test hangs on pami/pamilrts ppc64le. |
| #1787 | 2018 | good first issue,AMPI,documentation,openmp integration | AMPI+OpenMP documentation & example | A |  | AMPI+OpenMP documentation and example. |
| #1795 | 2018 | cleanup | Remove cc-cray 128 bit defines | I |  | cc-craycc.h still disables __int128 (lines 1-5). Retest with CCE (relevant for Frontier) and drop the defines if they are no longer needed. |
| #1796 | 2018 | Bug | Support for partitions on netlrts-linux-x86_64-tcp builds | O |  | +partitions on netlrts-tcp does not exit (netlrts/charmrun). |
| #1814 | 2018 | feature | Add CkLoop Split Execution | I |  | CkLoop split execution (CkLoop_Start/CkLoop_Wait). No such API on the reviewed line. Feature. |
| #1817 | 2018 | feature | Enable building charm with link time optimization | I |  | Build Charm with LTO. PR #3721 merged some fixes; no CMake LTO option yet. |
| #1818 | 2018 | feature,AMPI | Make AMPI's within-process messaging visible to Projections | A |  | Make AMPI in-process memcpy messaging visible to Projections (and to LB comm stats). |
| #1826 | 2018 | documentation | Add +showcpuaffinity option to the charm++ manual | K |  | Reconverse implements +showcpuaffinity (cpuaffinity.cpp:605) but doc/ has no mention of it. Documentation gap. |
| #1833 | 2018 | cleanup | Cleanup SDAG Closure refnum handling with enable_if | F |  | Done by a619eef5c ("simplify sdag refnum setting using C++11 std::enable_if"). sdag.h:25-29 uses enable_if. |
| #1836 | 2018 | cleanup | Cleanup warnings in charmdebug-python | I |  | ICC warnings in charmdebug-python and generated code (unused `impl_obj`). Still built when Python is found (ck-libs/CMakeLists.txt:187). Cosmetic. |
| #1837 | 2018 | Bug | uFcontext thread issue on ARM 64 bit systems | O |  | uFcontext ULTs on ARM64. The reviewed line uses reconverse threads on boost-context, not QuickThreads/uFcontext. |
| #1844 | 2018 | feature | SMP/non-SMP agnostic job launching arguments | F |  | Reconverse has no SMP/non-SMP distinction: always SMP, one `+pe <total>` argument, and launch goes through lcrun/srun. |
| #1859 | 2018 | Bug,AMPI | Megampi hangs on multicore-linux-x86_64 with +CmiSleepOnIdle | A |  | Megampi hangs on multicore with +CmiSleepOnIdle. |
| #1860 | 2018 | feature | Support HostBuffer shared memory allocation of one buffer per physical host at same address on all hosts | I |  | HostBuffer: one shared buffer per physical host at the same address. Feature. |
| #1862 | 2018 | feature | Simplify custom reducer registration | I |  | Simplify custom reducer registration. CkReduction::addReducer (ckreduction.h:252) still needs registration at init. Feature. |
| #1865 | 2018 | feature | Implement zero copy translation for move semantics and rvalue refs | I |  | Zerocopy translation for rvalue/const parameters (move semantics). Feature. |
| #1868 | 2018 | feature | Implement move semantics for object migration | I |  | Move semantics / RDMA get for object migration. Feature. |
| #1876 | 2018 | feature | Use IP multicast for faster broadcast and multicast on netlrts | O |  | IP multicast for netlrts/verbs broadcast. |
| #1878 | 2018 | feature | Tracing/Projections support for visualizing the communication graph | I |  | Projections communication-graph view. Feature. |
| #1880 | 2018 | Bug,AMPI | AMPI should reference count MPI_Op's | A |  | AMPI must reference-count MPI_Op objects. |
| #1898 | 2018 | cleanup | C++ cleanup of code formerly compiled as C | O |  | C++ cleanup of the former C sources in classic conv-core and machine layers. That code is not built and is slated for deletion. |
| #1904 | 2018 | Bug,smp | Review CMK_PCQUEUE_LOCK and CMK_NO_ASM_AVAILABLE | O |  | CMK_PCQUEUE_LOCK and the inline-asm PCQueue in classic conv-core. Reconverse uses its own queues. |
| #1918 | 2018 | feature | Enable immediate method tracing when CMK_SMP_TRACE_COMMTHREAD is enabled | O |  | Tracing immediate methods under CMK_SMP_TRACE_COMMTHREAD. Reconverse has no comm thread. |
| #1924 | 2018 | Bug | Calls to chare array element entry methods can fail from [immediate] node group methods | O |  | Array sends from [immediate] methods fail on the comm thread (no location manager). Reconverse has no comm thread, and CmiBecomeImmediate is a no-op (converse.h:859). |
| #1931 | 2018 | feature | Direct API for pamilrts-linux-ppc64le | O |  | Direct API for pamilrts. |
| #1939 | 2018 | feature,provisioning | Add concept of exclusions to automated provisioning arguments | O |  | Provisioning exclusion arguments (++excludeCoresPerHost, ...) for charmrun/+autoProvision. Not on the reviewed line; launchers (srun) and +pemap cover this. |
| #1940 | 2018 | Bug,smp | Singleton chare and nodegroup creation hangs with non-bitvec queues in SMP mode | O |  | The hang is in classic queueing.c (SMP with int prio and randomized queue). Note: the CMake RANDOMIZED_MSGQ option (CMakeLists.txt:183) is now a no-op, because reconverse never reads CMK_RANDOMIZED_MSGQ. |
| #1942 | 2018 | Bug,CharmDebug | CkStartQD never triggered even though all entry methods have returned | D |  | CkStartQD callback never fires (verbs, 1 PE). No reproducer was ever given. Needs a test case to tell whether shared QD code is at fault. |
| #1949 | 2018 | Bug | Ensure that 'End of Program' message is printed consistently for every charm program execution | O |  | "End of program" not printed on verbs-smp exit (classic machine exit path). |
| #1955 | 2018 | Bug | tests/charm++/chkpt hangs for mpi-win-x86_64-smp  | O |  | chkpt test hangs on mpi-win-x86_64-smp (MPI layer, Windows). |
| #1956 | 2018 | Bug | tests/charm++/sdag/migration and tests/charm++/sdag/anytimeMigration fail on mpi-win-x86_64-smp with debug options (-g -O0) | O |  | sdag migration tests fail on mpi-win-x86_64-smp debug. |
| #1959 | 2018 | Bug | examples/charm++/TRAM/randomAccessGroup crashes on mpi-win-x86_64-smp with debug options (-g -O0) | O |  | TRAM randomAccessGroup crash on mpi-win-x86_64-smp debug. |
| #1963 | 2018 | feature,AMPI | AMPI implements subcommunicators in an unscalable fashion | A |  | AMPI subcommunicators use one chare array each, so the number of communicators is limited by the collection bits in the id. |
| #1974 | 2018 | feature,smp | nocopy accelerated section multicast | I |  | Nocopy/RDMA section multicast with one copy per address space. Optimization. |
| #1976 | 2018 | Bug | ci files with extensive  code blocks result in incorrect line numbering in syntax errors and in gdb | D |  | charmxi line numbers wrong after large serial blocks (#line emitted at Serial.C:58, xi-util.C:228). The only reproducer is NDA code; needs a public test case. |
| #1986 | 2018 | feature | Avoid msg creation when contributing a pointer to a streamable reduction | I |  | Streamable reduction contributing into one per-PE message. Needs a new API. |
| #1991 | 2018 | Bug | CkScanf broken with charmrun | O |  | CkScanf fails under charmrun (charmrun input_scanf_chars). Reconverse CmiScanf is plain vscanf (convcore.cpp:1146). |
| #1992 | 2018 | good first issue,feature | Distribute a Valgrind error suppression file that suppresses known RTS memory leaks | I |  | Ship a Valgrind suppression file for known RTS leaks. None in the tree. |
| #1997 | 2018 | cleanup,Build & test automation | Improve coverage of tests for ck.C | I |  | ck.C test coverage. The coverage campaign already works on this. |
| #1998 | 2018 | cleanup,Build & test automation | integrate charmpy tests into coverage | I |  | charm4py (ext API) in the coverage runs. |
| #2000 | 2018 | cleanup,Build & test automation | debug and introspection API coverage | I |  | Coverage for the debug/introspection (CCS) API. |
| #2016 | 2018 | Bug,machine layers,provisioning | +autoProvision ignores numactl settings | O |  | +autoProvision ignores numactl (classic provisioning). Reconverse takes an explicit +pe count. |
| #2017 | 2018 | feature | Support for freeing Sections | K |  | No user API to free a section and its CkMulticastMgr state: teardown/freeup (ckmulticast.ci:18-19) are internal and undocumented. Apps that create many sections keep that memory; the AMPI port needs this. |
| #2018 | 2018 | Bug | Use of function pointers causes CkCallback errors in some ASLR environments | K | * | ckcallback.C:509-513 pups raw function pointers for call1Fn/callCFn. Across processes under ASLR/PIE (the default on Linux, launched with lcrun/srun and no setarch), these callbacks crash or abort. setReductionClient still uses callCFn (ckreduction.C:955). |
| #2024 | 2018 | Bug,charmxi | Perfect forwarding support is broken for templated entry methods | D |  | Perfect-forwarding defaults (xi-Entry.C:474) cannot deduce template parameters that appear only in a nested type (`typename Tag::x&`). Plain C++ cannot deduce these either. Explicit template arguments work around it; needs a call on whether charmxi should do more. |
| #2036 | 2018 | support | Ensure zero copy API transfers are included in CommLB | K |  | CkArray::recordSend (ckarray.C:1894,1931) records only envelope getTotalsize(), so zerocopy rget/rput payload bytes are invisible to LB communication stats (comm-aware LBs). |
| #2038 | 2018 | feature | Design a Many to Many API on the Zerocopy API | I |  | Many-to-many API built on zerocopy. Feature. |
| #2040 | 2018 | Bug,machine layers | pamilrts machine layer is less performant than pami machine layer | O |  | pamilrts slower than pami. |
| #2045 | 2019 | Bug | Chare array broadcast messages don't appear to be freed by the runtime | K |  | Array broadcasts are kept for anytime migration and freed only by spring cleaning on CcdPERIODIC_1minute (ckarray.C:540-541). anytime migration is on by default (init.C:383). A high broadcast rate grows memory for up to 1-2 minutes; a count- or size-based trigger was proposed. |
| #2048 | 2019 | Bug | Examine the converse header fields for all layers and remove/reduce fields wherever applicable | O |  | Shrink converse header fields per machine layer (classic layers). A reconverse header review would be a separate item. |
| #2051 | 2019 | Bug | QD and AtSync may not work well together | D |  | QD reported instantly after AtSync. Unknown whether LB-framework messages are invisible to QD; needs a reproducer. |
| #2061 | 2019 | feature | Support parameter marshaling for section broadcast | K |  | Section multicast via CkMulticast still requires CkMcastBaseMsg messages and no parameter marshalling (manual.rst:6977-6995). Common usability limitation. |
| #2064 | 2019 | cleanup | Use 64bit ID for singleton chares | I |  | 64-bit IDs for singleton chares (CkChareID still carries an object pointer). Fold into the id redesign #3994. |
| #2065 | 2019 | cleanup | Use 64bit ID for groups/nodegroups | I |  | 64-bit IDs for groups/nodegroups. Fold into #3994. |
| #2066 | 2019 | cleanup | Use 64bit IDs for tracing | I |  | 64-bit IDs in tracing. Fold into #3994. |
| #2067 | 2019 | feature | Add support for variable-sized messages to TRAM | I |  | Variable-sized messages in TRAM. Feature. |
| #2072 | 2019 | feature,charmxi | Improve variable-sized message declarations | I |  | Better variable-sized message declaration syntax in .ci (API break). Feature. |
| #2073 | 2019 | cleanup | Evaluate merging lib/ and lib_so/ folders | I |  | Merge lib/ and lib_so/. The CMake build still writes shared libs to lib_so/ (CMakeLists.txt:226,463-471). Cleanup. |

## Counts

| bucket | count |
|---|---|
| K | 8 |
| F | 4 |
| O | 31 |
| A | 12 |
| I | 35 |
| D | 7 |
| total | 97 |

## K items, ranked (critical first)

1. **#2018 (critical)**: CkCallback call1Fn/callCFn serialize raw function pointers. In multi-process runs without ASLR disabled (the lcrun/srun default), a callback that crosses processes crashes or aborts, and the legacy setReductionClient path still relies on it. Fix: a registered function table, or forbid these callback types across processes.
2. **#2045**: With default array options (anytime migration on), broadcast messages are kept until the 1-minute spring-cleaning timer runs. High-rate broadcasts can therefore grow memory substantially. Replace the timer with a count/size trigger, or change the defaults.
3. **#2036**: LB communication stats count only envelope bytes, so zerocopy payloads are invisible to comm-aware load balancers. Their decisions are wrong for zerocopy-heavy applications.
4. **#1700**: Overloaded [reductiontarget] entry methods produce a charmxi redeclaration compile error, because `redn_wrapper_<name>` has no overload suffix.
5. **#2061**: Section multicast through CkMulticast requires CkMcastBaseMsg messages and cannot use parameter marshalling. This is a common usability limitation.
6. **#2017**: Users have no API to free a section or its CkMulticastMgr tree state. Programs that create many dynamic sections keep that memory, and the AMPI port needs this.
7. **#1545**: PUP of `std::vector<T, A>` with a custom allocator does not compile. A two-line template change fixes the stateless case.
8. **#1826**: The manual does not document `+showcpuaffinity`, which reconverse implements.

## Side findings

- The `RANDOMIZED_MSGQ` CMake option (CMakeLists.txt:183) does nothing on the reviewed line, because reconverse never reads `CMK_RANDOMIZED_MSGQ`. Several tests still `#if` on it (found under #1940).
- `src/libs/conv-libs/openmp_llvm` (135 files) is not in the CMake build, so the OpenMP integration is effectively not shipped (affects #1586, #1666, #1787). It needs a keep-or-drop decision.
- The `examples/charm++/state_space_searchengine` Makefile refers to a `searchengineLib` directory that does not exist (#1550).
