# Issue audit, slice 2 (97 issues, #931 to #1538)

This audit asks whether each open charmplusplus/charm issue still matters on `reviewed-with-reconverse` (tip ebdc49c66, reconverse submodule b30ad31), audited 2026-09-30. Code checks were run against that branch. The charmxi and runtime reproductions used the local build `~/software/recharm/charm/reconverse-darwin-arm8`, which was built from that branch.

Buckets: K = keep, F = fixed or superseded, O = obsolete on the reviewed line, A = AMPI/TCharm port list, I = wishlist, D = needs a decision. A star (*) marks a critical item.

| issue | year | labels | bucket | critical? | reason / evidence |
|---|---|---|---|---|---|
| [#931](https://github.com/charmplusplus/charm/issues/931) avoid failing back to hostname ip lookup for physical node detection | 2015 | Bug | K |  | Physical-node identity is still keyed only on the IP address. Reconverse `skt_my_ip()` (contrib/reconverse src/conv-topology.cpp) now tries hostname resolution FIRST and does not filter loopback, so hosts whose /etc/hosts maps the hostname to 127.0.1.1 all collapse into one "physical node". That affects topology, LB, and (with CMK_USE_SHMEM, off by default) the IPC path. |
| [#952](https://github.com/charmplusplus/charm/issues/952) Update AMPI's version of ROMIO | 2016 | feature,AMPI | A |  | ROMIO update: AMPI. |
| [#969](https://github.com/charmplusplus/charm/issues/969) AMPI support for collectives on inter-communicators | 2016 | feature,AMPI | A |  | Intercommunicator collectives: AMPI. |
| [#971](https://github.com/charmplusplus/charm/issues/971) Improve malloc implementation | 2016 | feature | I |  | Malloc strategy review, no defect. The ugni/PAMI references are obsolete, but the question applies to reconverse's allocator. |
| [#980](https://github.com/charmplusplus/charm/issues/980) Cleanup examples/charm++/cuda/hello | 2016 | cleanup,GPU support | F |  | examples/charm++/cuda/hello/Makefile now uses `CUDATOOLKIT_HOME ?=` and `CUDA_ARCH ?=` instead of a hard-coded NVCC path. |
| [#1000](https://github.com/charmplusplus/charm/issues/1000) +setcpuaffinity places replicas on same host on same cores | 2016 | Bug,machine layers | K |  | Reconverse `set_auto_affinity()` (src/cpuaffinity.cpp) counts only the ranks of its own partition (`Cmi_numnodes` is set to the partition size in CmiCreatePartitions), so +replicas on one host still pick the same PUs. `+pemap` uses CmiMyPeGlobal and is fine. Found by reading the code; not run. |
| [#1003](https://github.com/charmplusplus/charm/issues/1003) "-memory paranoid" doesn't work with smp build | 2016 | Bug | O |  | `-memory paranoid` is a classic conv-core memory module. Reconverse builds ship no libmemory-* at all. |
| [#1005](https://github.com/charmplusplus/charm/issues/1005) Some Projections views broken for AMPI with user-registered functions | 2016 | Bug,tracing | A |  | Projections views for AMPI user-registered functions. |
| [#1006](https://github.com/charmplusplus/charm/issues/1006) charmxi: nested braces in serial blocks can lead to erroneous syntax errors | 2016 | Bug,charmxi | K |  | Reproduced on the reviewed build: a `serial { {{ } } }` block gives the same spurious parse error, and then charmxi aborts with std::length_error (rc 134). |
| [#1009](https://github.com/charmplusplus/charm/issues/1009) Improve Parsing for SDAG Boolean Expressions | 2016 | Bug,charmxi | I |  | SDAG syntax sugar for complex boolean conditions. |
| [#1035](https://github.com/charmplusplus/charm/issues/1035) Idle PEs compete with comm thread for node queue lock | 2016 | Bug,machine layers | O |  | Node-queue lock contention between idle PEs and the comm thread (classic machine-common-core.C). Reconverse has no comm thread and its own queues. |
| [#1037](https://github.com/charmplusplus/charm/issues/1037) megacon link failure | 2016 | Bug | O |  | megacon link failure with tracing on classic multicore/smp (TraceTimerCommon in machine.C). |
| [#1039](https://github.com/charmplusplus/charm/issues/1039) reject pemap/commap with duplicate or too few cpus | 2016 | feature,machine layers,provisioning | I |  | Validation of +pemap. Reconverse `search_pemap()` still accepts duplicate or too-short maps silently. Nice-to-have warning. |
| [#1040](https://github.com/charmplusplus/charm/issues/1040) support multiple InfiniBand cards per node | 2016 | feature,machine layers | O |  | Multiple InfiniBand HCAs in the verbs layer. Reconverse leaves NIC selection to LCI. |
| [#1044](https://github.com/charmplusplus/charm/issues/1044) AMPI support for MPI_THREAD_SERIALIZED | 2016 | feature,AMPI | A |  | MPI_THREAD_SERIALIZED: AMPI. |
| [#1056](https://github.com/charmplusplus/charm/issues/1056) make trivial SDAG nodegroup entries reentrant | 2016 | feature | I |  | SDAG entries on nodegroups are still made [exclusive] even when they are serial-only (xi-Entry.C preprocessSDAG adds SLOCKED). A design choice; making them reentrant is optional. |
| [#1059](https://github.com/charmplusplus/charm/issues/1059) Unify Data Collection in Charm++ | 2016 | cleanup,tracing | I |  | Unify data collection across LB, PICS, and tracing. Refactor wish. |
| [#1060](https://github.com/charmplusplus/charm/issues/1060) AMPI compliance with MPI-3.1 standard | 2016 | feature,AMPI | A |  | MPI-3.1 compliance tracker: AMPI. |
| [#1063](https://github.com/charmplusplus/charm/issues/1063) Error in buddy assignment code for checkpointing | 2016 | Bug,fault tolerance | D |  | The per-element buddy assignment flaw is in the `!CMK_CHKP_ALL` path, which is compiled out (ckmemcheckpoint.C hard-codes `#define CMK_CHKP_ALL 1`). Design call: delete that mode or fix it. |
| [#1065](https://github.com/charmplusplus/charm/issues/1065) Create a more efficient caching structure for location lookup | 2016 | cleanup | I |  | CkLocMgr still uses std::unordered_map (cklocation.h). A faster or caching structure is optional and overlaps the queued process-level location cache review. |
| [#1095](https://github.com/charmplusplus/charm/issues/1095) Improve AMPI error handling | 2016 | feature,AMPI | A |  | AMPI error-handling consistency. |
| [#1100](https://github.com/charmplusplus/charm/issues/1100) Cannot create variable sized messages with objects that have virtual methods | 2016 | Bug | K |  | Varsize messages still lay out array members by casting offsets into the buffer; no constructors run, so classes with virtual methods have no vtable and crash. Limitation users can hit. |
| [#1101](https://github.com/charmplusplus/charm/issues/1101) Poor error messages when forgetting to 'extern module x' in ci files | 2016 | Bug | F |  | Now fails cleanly: tested on reconverse +pe 2, it aborts with "Did you forget to import a module...?" (b030a2976, Cleanup #2054). |
| [#1105](https://github.com/charmplusplus/charm/issues/1105) AMPI support for large counts | 2016 | feature,AMPI | A |  | MPI large-count support: AMPI. Depends on #1378. |
| [#1115](https://github.com/charmplusplus/charm/issues/1115) verbs slower than mpi on Omni-Path | 2016 | Bug,machine layers | O |  | verbs vs MPI performance on Omni-Path. |
| [#1117](https://github.com/charmplusplus/charm/issues/1117) Shrink-expand malleable jobs should be able to change node set | 2016 | feature,migration,shrink-expand | I |  | Shrink/expand onto new, previously unknown nodes. Shrink/expand exists via checkpoint; this extends it. |
| [#1135](https://github.com/charmplusplus/charm/issues/1135) Pass a future into or return one from [sync] entry methods, so that the caller isn't forced to block | 2016 | feature,charmxi | I |  | Futures for [sync] entry methods. Only [iget] (array elements) returns a CkFutureID today. |
| [#1144](https://github.com/charmplusplus/charm/issues/1144) Batched message delivery to objects for better cache behavior | 2016 | feature | I |  | Batched delivery of messages to one object for cache locality. |
| [#1158](https://github.com/charmplusplus/charm/issues/1158) AMPI scatter(v) performance is poor | 2016 | feature,AMPI | A |  | AMPI scatter(v) performance. |
| [#1159](https://github.com/charmplusplus/charm/issues/1159) AMPI should check for truncated messages | 2016 | Bug,AMPI | A |  | AMPI MPI_ERR_TRUNCATE handling. |
| [#1167](https://github.com/charmplusplus/charm/issues/1167) Scripted tests for all mini-apps | 2016 | support,Build & test automation | I |  | Scripted tests for external mini-apps. Overlaps the planned apps-ci.yaml. |
| [#1173](https://github.com/charmplusplus/charm/issues/1173) Automatic process launching, thread spawning, and hardware binding | 2016 | feature,provisioning | O |  | hwloc-based ++numHosts/++ppn launching implemented in charmrun for netlrts/verbs. Launch is now lcrun/srun plus reconverse affinity. |
| [#1177](https://github.com/charmplusplus/charm/issues/1177) Support variable numbers of processes per host, each with identical thread counts | 2016 | feature,provisioning | O |  | Variable processes per host under charmrun provisioning (XE/XK). Launcher-driven now. |
| [#1179](https://github.com/charmplusplus/charm/issues/1179) Support automated launch/spawn/bind on gni-crayxc/gni-crayxe systems | 2016 | feature,provisioning | O |  | Automated launch/bind for gni-crayxc/gni-crayxe. |
| [#1182](https://github.com/charmplusplus/charm/issues/1182) Support automated thread spawn/bind for MPI SMP builds | 2016 | feature,provisioning | O |  | Automated thread spawn/bind for MPI SMP builds. |
| [#1183](https://github.com/charmplusplus/charm/issues/1183) megatest and megacon should work for large node counts | 2016 | Bug,Build & test automation | I |  | megatest runtime scaling at large node counts. megatest lives on (tests/charm++/megatest); megacon does not. Test-suite improvement. |
| [#1202](https://github.com/charmplusplus/charm/issues/1202) Memory leaks in converse's cldb | 2016 | Bug | O |  | Valgrind leaks in classic conv-ldb/cldb.C and conv-core/cputopology.C. Neither is built on reconverse, which has its own cldb.cpp and conv-topology.cpp. |
| [#1205](https://github.com/charmplusplus/charm/issues/1205) tlsglobals on Linux-x86_64 only works with GCC and Clang | 2016 | Bug,AMPI | A |  | -tlsglobals with non-GCC/Clang compilers: AMPI privatization. |
| [#1227](https://github.com/charmplusplus/charm/issues/1227) Support template entry methods in generated code from TRAM [aggregate] entry method attribute | 2016 | Bug | K |  | Reproduced: a templated `[aggregate]` entry method passes charmxi, but the generated decl.h does not compile ("use of undeclared identifier 'T'", plus proxy-type errors). |
| [#1232](https://github.com/charmplusplus/charm/issues/1232) AMPI migration detection mechanisms | 2016 | feature,AMPI | A |  | AMPI migration-detection API. |
| [#1245](https://github.com/charmplusplus/charm/issues/1245) Support for [accel] entry methods | 2016 | feature | O |  | [accel] entry methods: dead feature. The grammar tokens remain, but no accelerator manager runtime exists in src/. |
| [#1246](https://github.com/charmplusplus/charm/issues/1246) Versioning for [accel] entry methods | 2016 | feature | O |  | Versioning for [accel] entry methods: same dead feature. |
| [#1247](https://github.com/charmplusplus/charm/issues/1247) Problem with reduction when inheriting pure virtual functions | 2016 | Bug | K |  | Reproduced on reconverse: with `class B : public C, public CBase_B` (C a polymorphic non-chare), `+pe 2` segfaults in CkReductionMgr::RecvMsg -> CkQ::enq, because the object pointer is cast through void* to a base at a nonzero offset. `+pe 1` passes; listing CBase_B first avoids it. |
| [#1248](https://github.com/charmplusplus/charm/issues/1248) Charm++ scatter(v) collective interface | 2016 | feature | I |  | Charm++ scatter(v) collective. None exists in ck-core. |
| [#1250](https://github.com/charmplusplus/charm/issues/1250) Tuple reductions cannot have [reductiontarget] callbacks | 2016 | Bug | I |  | `[reductiontarget]` unmarshalling for tuple reductions. charmxi has no tuple support; the CkReductionMsg::toTuple workaround exists. |
| [#1256](https://github.com/charmplusplus/charm/issues/1256) Document '-tracemode perfReport' in the manual | 2016 | documentation | K |  | `-tracemode perfReport` still exists (ck-pics/trace-perf.C, charmc) but the manual does not document it. Doc gap. |
| [#1258](https://github.com/charmplusplus/charm/issues/1258) AMPI support for MPI-3 RMA | 2016 | feature,AMPI | A |  | MPI-3 RMA: AMPI. |
| [#1260](https://github.com/charmplusplus/charm/issues/1260) MPI_T tools interface support in AMPI | 2016 | feature,AMPI | A |  | MPI_T tools interface: AMPI. |
| [#1264](https://github.com/charmplusplus/charm/issues/1264) Use Infiniband multicast for broadcast/multicast | 2016 | feature,machine layers | O |  | InfiniBand hardware multicast in a machine layer. |
| [#1279](https://github.com/charmplusplus/charm/issues/1279) Proactive fault tolerance fails due to sending message to dead node. | 2016 | Bug,fault tolerance | O |  | Proactive fault tolerance and evacuation. The CmiNodeAlive/allowMessagesOnly code no longer exists in ck-core; FT is a tombstone. |
| [#1300](https://github.com/charmplusplus/charm/issues/1300) Integrated OpenMP should co-exist with MPI interop | 2016 | feature,mpi interoperability,openmp integration | O |  | Integrated LLVM OpenMP runtime (conv-libs/openmp_llvm) is not built by any CMake target; MPI interop is also untested on reconverse. |
| [#1303](https://github.com/charmplusplus/charm/issues/1303) Implement MPI-R debugging hooks to support Allinea DDT and Rogue Wave Totalview | 2016 | feature | I |  | MPIR debugger hooks. With srun/lcrun launch the launcher provides MPIR, so this is optional. |
| [#1304](https://github.com/charmplusplus/charm/issues/1304) Charmrun: implement MPI-R debugging hooks | 2016 | feature | O |  | MPIR hooks in charmrun. |
| [#1305](https://github.com/charmplusplus/charm/issues/1305) Implement MPI-R debugging hooks in RTS startup | 2016 | feature | I |  | MPIR hooks in RTS startup. Same as #1303. |
| [#1306](https://github.com/charmplusplus/charm/issues/1306) PythonCCS-client needs to be compiled without CONVERSE | 2016 | Bug | O |  | PythonCCS client needs -DCMK_NOT_USE_CONVERSE. PythonCCS builds only with Python 2, and reconverse has no CCS. |
| [#1309](https://github.com/charmplusplus/charm/issues/1309) Use CkMulticast for collectives on AMPI subcommunicators | 2016 | feature,AMPI | A |  | CkMulticast for AMPI subcommunicator collectives. |
| [#1315](https://github.com/charmplusplus/charm/issues/1315) examples/charm++/jacobi*d are non-exemplary HPC code, using nested arrays | 2016 | good first issue,cleanup | I |  | The examples/charm++/jacobi*d nested arrays are gone, but the same pattern remains in examples/charm++/gaussSeidel3D and examples/charm++/topology/jacobi2d. Cleanup. |
| [#1317](https://github.com/charmplusplus/charm/issues/1317) Add Metabalancer tests to nightly build | 2016 | support,load balancing | F |  | Metabalancer test exists: tests/charm++/load_balancing/meta_lb_test (period_selection) is in the load_balancing Makefile. |
| [#1321](https://github.com/charmplusplus/charm/issues/1321) multiple communication threads per process | 2016 | feature,machine layers | O |  | Multiple comm threads per process. Reconverse has no comm thread. |
| [#1324](https://github.com/charmplusplus/charm/issues/1324) Collision Detection library failures due to changes in demand creation | 2016 | Bug | D |  | Collide failures after the 2017 demand-creation change. Collide is actively maintained (#3036, #3591, #3921), but nobody has re-run its tests since 2020. Needs a run of examples/collide on the reviewed line. |
| [#1328](https://github.com/charmplusplus/charm/issues/1328) AMPI shrink/expand example and documentation | 2016 | good first issue,feature,AMPI,shrink-expand | A |  | AMPI shrink/expand example and docs. |
| [#1333](https://github.com/charmplusplus/charm/issues/1333) Eliminate need for .ci files | 2016 | feature | I |  | Eliminate .ci files. |
| [#1334](https://github.com/charmplusplus/charm/issues/1334) Chare and entry method registration, instantiation, and invocation without code generated by charmxi from .ci files | 2016 | feature | I |  | Registration without charmxi. |
| [#1335](https://github.com/charmplusplus/charm/issues/1335) Replace SDAG in charmxi with pure C++ | 2016 | feature | I |  | SDAG in pure C++. |
| [#1337](https://github.com/charmplusplus/charm/issues/1337) Cpv Declarations of types with constructors may induce 'static initialization order fiasco' | 2016 | Bug | I |  | ckcallback fixed (now a pointer). Remaining: pose.C `CpvDeclare(eventID, ...)` and benchmarks/xcastredn. Low value. |
| [#1340](https://github.com/charmplusplus/charm/issues/1340) Applications should only pay time and memory cost of features used | 2016 | feature | I |  | Pay-for-what-you-use features. |
| [#1364](https://github.com/charmplusplus/charm/issues/1364) Review use of volatile variables in the runtime | 2017 | Bug,smp | I |  | The volatile list is mostly classic machine layers. Remaining on the reviewed line: CkLoop.C `volatile int gCrtCnt/exitFlag` (used with __sync ops). Small cleanup. |
| [#1371](https://github.com/charmplusplus/charm/issues/1371) Within-node PUP API | 2017 | feature,smp | I |  | Within-node PUP API for intra-process migration. |
| [#1378](https://github.com/charmplusplus/charm/issues/1378) 64-bit Charm message sizes | 2017 | feature | K | * | `envelope::totalsize` is `unsigned int` (envelope.h), `_allocEnv` computes it in UInt with no overflow check, and reconverse `CmiAlloc(int size)` takes int. A message over 2 GB silently wraps into an undersized buffer: memory corruption with no diagnostic. |
| [#1386](https://github.com/charmplusplus/charm/issues/1386) Allow destroying Arrays, Groups, and NodeGroups | 2017 | feature | I |  | Destroying groups and nodegroups (and whole arrays). ckDestroy exists only on array elements. |
| [#1387](https://github.com/charmplusplus/charm/issues/1387) Optimised algorithms for scatterv | 2017 | support | I |  | Optimized scatterv algorithms. |
| [#1394](https://github.com/charmplusplus/charm/issues/1394) Node-level message aggregation for CkMulticast | 2017 | feature | I |  | Nodegroup-level CkMulticast (still a group in ckmulticast.ci). |
| [#1401](https://github.com/charmplusplus/charm/issues/1401) Converting OpenMP test suite for the OpenMP integration. | 2017 | feature,openmp integration | O |  | Test suite for the integrated OpenMP runtime, which is not built on the reviewed line. |
| [#1414](https://github.com/charmplusplus/charm/issues/1414) Autobuild should run tests on SMP builds with multiple threads per process | 2017 | support,Build & test automation | F |  | reconverse-ci.yaml runs the test dirs multi-threaded and with TESTRUN_PROCS=2 over lcrun. Autobuild is gone. |
| [#1422](https://github.com/charmplusplus/charm/issues/1422) Cleanup dangling issues from 64bit merge | 2017 | cleanup | I |  | 64-bit merge leftovers: flushLocalRecs/reclaimRemote still exist in cklocation.C. Cleanup. |
| [#1426](https://github.com/charmplusplus/charm/issues/1426) AMPI F08 bindings | 2017 | feature,AMPI | A |  | AMPI Fortran 2008 bindings. |
| [#1427](https://github.com/charmplusplus/charm/issues/1427) Virtualize handling of Fortran IO units | 2017 | feature,AMPI | A |  | Fortran IO unit virtualization for AMPI. |
| [#1428](https://github.com/charmplusplus/charm/issues/1428) AMPI TLS privatization support for IBM POWER | 2017 | feature,AMPI | A |  | AMPI TLS privatization on POWER. |
| [#1431](https://github.com/charmplusplus/charm/issues/1431) Charmxi runs out of memory | 2017 | Bug,charmxi | F |  | Not reproducible: 6 templated entry methods x 10 instantiations run through charmc in 0.15 s and 20 MB RSS. |
| [#1435](https://github.com/charmplusplus/charm/issues/1435) Collapse prioritized msg buckets into priority queue | 2017 | feature | O |  | Collapse prioritized buckets in classic conv-core msgq.h. Not used by reconverse's scheduler. |
| [#1440](https://github.com/charmplusplus/charm/issues/1440) smp pes sending messages still block due to other send activity | 2017 | Bug,machine layers | O |  | SMP send contention in verbs/net-ibverbs. |
| [#1449](https://github.com/charmplusplus/charm/issues/1449) AMPI support for MPI_Win_allocate_shared | 2017 | feature,AMPI | A |  | MPI_Win_allocate_shared: AMPI. |
| [#1459](https://github.com/charmplusplus/charm/issues/1459) Zero-copy send support for the netlrts machine layer | 2017 | feature,machine layers | O |  | Zero-copy send for netlrts. |
| [#1465](https://github.com/charmplusplus/charm/issues/1465) Spanning Tree implementation for scatterv | 2017 | feature | I |  | Spanning-tree scatterv. Same as #1387. |
| [#1471](https://github.com/charmplusplus/charm/issues/1471) Parallel Prefix No Barrier Example in Charm Tutorial Hangs on MPI Layer | 2017 | Bug | O |  | Tutorial hang on the MPI machine layer, never reproduced. |
| [#1476](https://github.com/charmplusplus/charm/issues/1476) Fix Make.depends for libraries | 2017 | cleanup | O |  | Make.depends for libraries. CMake is the only build system. |
| [#1477](https://github.com/charmplusplus/charm/issues/1477) All Load Balancing Strategies should be CPU frequency (rate) aware | 2017 | Bug | I |  | CPU-frequency-aware LB strategies (pe_speed exists only as a static field). |
| [#1485](https://github.com/charmplusplus/charm/issues/1485) CharmDebug in SMP mode does not work | 2017 | feature,CharmDebug | D |  | CharmDebug in SMP mode. Reconverse is always threaded, but it has no CCS (the ledger marks conv-ccs needs-judgment), so CharmDebug does not run at all. Relevant only if CCS/CharmDebug is ported. |
| [#1497](https://github.com/charmplusplus/charm/issues/1497) CMA support for passing data between processes on the same node | 2017 | feature | O |  | CMA for intra-node transfers in classic machine layers. Reconverse has its own shm/xpmem IPC. |
| [#1498](https://github.com/charmplusplus/charm/issues/1498) SDAG methods are not properly inherited by chare subclasses | 2017 | Bug | K |  | Reproduced: `chare B : A` with `when foo()` on a parent-declared foo gives "no matching declaration for entry method 'foo(void)'". |
| [#1499](https://github.com/charmplusplus/charm/issues/1499) Remove the need to declare entry method parameters as "nocopy" | 2017 | feature,charmxi | I |  | Drop the need for the `nocopy` keyword. |
| [#1500](https://github.com/charmplusplus/charm/issues/1500) Entry Methods Always Take lvalue References (feature/bug) | 2017 | Bug,charmxi | F |  | Rvalue references and perfect forwarding merged (per epmikida 2019). Issue left open. |
| [#1510](https://github.com/charmplusplus/charm/issues/1510) Hang in tests/charm++/chkpt when using -tracemode perfReport | 2017 | Bug,tracing | K |  | `traceAutoPerfExitFunction` still blocks on `endStepResumeCb(..., CkCallbackResumeThread())` (picsautoperf.C:734), and after +restart getTraceOn() is false, so the callback never fires. Hang with perfReport plus restart; TRACING builds only. |
| [#1520](https://github.com/charmplusplus/charm/issues/1520) multicore-darwin-x86_64 megatest hangs when built with --enable-randomized-msgq --with-prio-type=int --enable-error-checking -debug | 2017 | Bug | O |  | Hang with --enable-randomized-msgq on classic multicore. CMK_RANDOMIZED_MSGQ exists only in conv-core msgq.h. |
| [#1521](https://github.com/charmplusplus/charm/issues/1521) CkIO file read support | 2017 | feature,documentation | F |  | CkIO read exists (Ck::IO::startReadSession/read in ckio.h, tests/charm++/io_read) and is documented in doc/libraries/manual.rst. |
| [#1536](https://github.com/charmplusplus/charm/issues/1536) AMPI interface for mapping ranks to worker threads | 2017 | feature,AMPI | A |  | AMPI rank-to-PE mapping interface. |
| [#1538](https://github.com/charmplusplus/charm/issues/1538) Support Shrink/Expand in verbs | 2017 | Bug,Verbs,shrink-expand | O |  | Shrink/expand for the verbs layer. |

## Counts

| bucket | count |
|---|---|
| K | 10 (1 critical) |
| F | 7 |
| O | 27 |
| A | 20 |
| I | 30 |
| D | 3 |
| total | 97 |

## K items, ranked

1. **#1378 (critical)**: messages over 2 GB silently corrupt memory. `envelope::totalsize` is an `unsigned int`, `_allocEnv` sums it with no overflow check, and reconverse `CmiAlloc` takes an `int`. A size guard that aborts cleanly is cheap and should go in before any real 64-bit or pipelined large-message design.
2. **#1247**: a chare or group whose C++ class lists a polymorphic non-chare base before `CBase_X` crashes on more than one PE. It segfaults in `CkReductionMgr::RecvMsg` because the object pointer is cast through `void*` to a base at a nonzero offset. Reproduced on reconverse with the reporter's files; `+pe 1` hides it.
3. **#931**: reconverse's `skt_my_ip()` resolves the hostname first and never rejects loopback. Nodes whose /etc/hosts maps the hostname to 127.0.1.1 all collapse into one physical node, which gives wrong topology for LB and TopoManager, and wrong IPC routing when CMK_USE_SHMEM is on.
4. **#1510**: a perfReport tracing build hangs at exit after `+restart`, because the `endStepResumeCb` resume-thread callback never fires once tracing is off. Only affects TRACING plus restart.
5. **#1000**: auto CPU affinity (`+setcpuaffinity` without `+pemap`) ranks PEs within a partition only, so `+replicas` sharing a host get the same PUs. Found by reading reconverse cpuaffinity.cpp; not run.
6. **#1227**: a templated `[aggregate]` (TRAM) entry method passes charmxi, but the generated decl.h does not compile ("undeclared identifier 'T'"). Reproduced.
7. **#1498**: SDAG `when` clauses in a derived chare cannot name an entry method declared in the parent ("no matching declaration"). Reproduced.
8. **#1006**: nested braces `{{` inside a serial block give a spurious parse error, and then charmxi aborts with std::length_error. Reproduced.
9. **#1100**: varsize message array members never have their constructors run, so element types with virtual methods have no vtable and crash on first use. Long-standing design limitation of message allocation.
10. **#1256**: `-tracemode perfReport` ships but is not documented in the manual.

## Side finding (not an issue in this slice)

Running `charmxi file.ci` directly segfaults, in `SerialConstruct::generateCode` -> `XStr::append(NULL)`, on any SDAG `serial` block with a non-empty body. The same file compiles fine through `charmc`, so the crash is probably a null source-file name when charmxi runs without charmc's options. This matters only for people who call charmxi by hand; it may deserve its own issue.

## Notes on method

- The reproductions for #1006, #1227, #1498, #1101, #1431 and #1247 were run with `bin/charmc` and `+pe N` on the reviewed-line build, in the session scratchpad.
- #1101 was reclassified from bug to F: the runtime now aborts with "Did you forget to import a module or instantiate a templated entry method in a .ci file?" (commit b030a2976).
- The O classification of classic memory modules, conv-ldb, msgq and conv-core rests on CMakeLists.txt, which skips `src/conv-core`, `src/QuickThreads` and the classic `ldb-*` libraries when `RECONVERSE` is set. The reconverse build's lib/ directory contains no libmemory-* or libconv-* files.
