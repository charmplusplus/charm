# Issue audit, slice 1 (charmplusplus/charm, reviewed-with-reconverse)

Audited 2026-09-30 against origin/reviewed-with-reconverse ebdc49c66 (reconverse pin b30ad31). Buckets: K keep, F fixed/superseded, O obsolete on the reviewed line, A AMPI/TCharm port list, I idea/wishlist, D needs a decision. Nothing was commented, closed or labelled.

| issue | year | labels | bucket | critical? | reason / evidence |
|---|---|---|---|---|---|
| #8 Ensure that various features work with partitions | 2013 | feature | D |  | No body; umbrella for "features with partitions". Reconverse implements partitions (converse.h:1306 CMK_HAS_PARTITION 1, tests/charm++/partitions); a concrete feature list is needed to act on it. |
| #9 Ensure that record-replay works with partitions | 2013 | feature | I |  | Record-replay works on reconverse (+record/+replay); no test combines it with +partitions. Candidate test to add. |
| #11 Make CCS work with partitions | 2013 | feature | D |  | CCS is not ported to reconverse (cmake/detect-features.cmake sets CMK_CCS_AVAILABLE 0 under RECONVERSE); moot until someone decides whether CCS returns. |
| #12 Factor out massive duplication in reductions | 2013 | cleanup | I |  | Refactor, no defect. Duplicate reduction logic still in ckreduction.C (CkReductionMgr, CkNodeReductionMgr) and ck-core/ckmulticast.C. |
| #13 CkCallback to section breaks | 2013 | feature | K |  | CkCallback(ep, sectionProxy) still stores raw pointers (s_section._elems, pelist in ckcallback.h:133-142; containsPointer() true for bcastSection), so pup/send off the creating PE fails; 2018 comment reports an assertion with 1D sections of 2D arrays. Same root cause as #235. |
| #16 Projections tracing with MPI Interoperability | 2013 | feature,mpi interoperability | D |  | Charm-MPI interoperation is known broken on reconverse (CharmBeginInit in mpi-interoperate.C aborts); tracing under interop is moot until interop is ported. |
| #19 Extend TopoManager to work on an n-dimensional torus | 2013 | feature | I |  | TopoManager is still built; n-D torus support is only relevant to Tofu-class machines. Low value. |
| #23 Optimization of MPI layer | 2013 | feature | O |  | MPI machine layer: not built on the reviewed line. |
| #28 Replica based FT | 2013 | feature | O |  | Replica-based fault tolerance: never merged; FT message-logging is a tombstone. |
| #34 Reduce Charm Message Send Overhead for Marshalled Messages | 2013 | feature | I |  | Marshalled-send overhead in the send path (ck.C/cklocation.C) is still there; the 2017 comment says zerocopy should cover the large-message case, but nobody has re-measured. |
| #41 LRTS Machine layer flow control | 2013 | feature,machine layers | O |  | LRTS machine-layer flow control: classic machine layers. |
| #44 Macports / Homebrew installation scripts | 2013 | support | I |  | The licensing objection is gone: LICENSE on the reviewed line is Apache-2.0. A MacPorts/Homebrew/Spack recipe for the reconverse build is optional. |
| #49 Define and document semantics of dynamic Insertion in Chare arrays wrt broadcasts and reductions | 2013 | feature | K |  | Semantics of dynamic insertion overlapping reductions/broadcasts is still undocumented: manual.rst:6677-6690 only says "different semantics - see the reduction manager". Phil's recipe (2015) and Kale's 2018 ask for a doc and test are not done. |
| #60 Exploit phase boundaries in iterative apps | 2013 | feature | I |  | Phase-boundary API: design idea, no defect. |
| #66 Execution within a power budget | 2013 | feature | I |  | Power-budget execution: research feature. |
| #69 Get large messages sent via the ZC API to be received directly using a GET from the new home. (rather than a GET from the old home and forwarding the large message) | 2013 | feature | I |  | ZC GET forwarding for migrated recipients: optimization for the ZC-API + LB case, deferred in 2019 as low impact. |
| #72 Better compression for Projections traces | 2013 | feature,tracing | I |  | Better compression for Projections logs (zstd/xz): optional. |
| #105 Unify memory pool implementations where possible | 2013 | feature | O |  | Pools named in the issue (cmipool, gni/pami mempools, arch/util/mempool.C) are not built on reconverse (converse.cmake:142-150); reconverse has its own src/mempool.C. Review that one as a new item if needed. |
| #106 Outgoing message prioritization | 2013 | feature | O |  | LRTS outgoing-message prioritization: classic machine layers. |
| #111 object location services: scalable cache pre-population | 2013 | feature | I |  | Location-cache pre-population after migration: optimization, deferred 2013. |
| #112 object location services: Share array element location cache above PE level | 2013 | feature,smp | I |  | Node/process-level location cache: not on the reviewed line (e417584dd is only on cupti_lb_reconverse); part of the queued location-manager overhaul. |
| #113 object location services: caching policies - expiration, size limits, replacement | 2013 | feature | I |  | CkLocCache::locMap (cklocation.h:349) still has no eviction or size bound. Memory growth only; no correctness issue. |
| #114 object location services: cache replication to avoid hotspots because of network-topological distance from homePE. | 2013 | feature | I |  | No body; cache replication for home-PE hotspots. Optimization. |
| #115 object location services: iteration over local objects | 2013 | feature | F |  | CkLocMgr::iterate(CkLocIterator&) exists (cklocation.h:313, 675). |
| #117 Projections traces with fewer files than 1-per-PE | 2013 | feature | I |  | Projections still writes one log per PE; +trace-subdirs and +traceprocessors mitigate. Node-level aggregation would be an enhancement. |
| #120 ckmulticast and any time migration | 2013 | Bug,migration | K |  | Hang/assert when CkMulticast section reductions are combined with anytime migration; ckmulticast.C has had no functional change since 2019; tests/charm++/delegation/multicast still exists. The new anytime_bcastred stress test covers array reductions, not section reductions. |
| #129 mempool code docs and review | 2013 | support | O |  | Review of classic mempool implementations; those pools are not built on reconverse. |
| #130 mempool unit and performance tests | 2013 | feature | O |  | Tests for the classic mempools; same as #129. |
| #150 Align -optimize flags with current compiler releases | 2013 | support | I |  | charmc -optimize still maps to per-compiler CMK_*_OPTIMIZE in src/arch/common/cc-*.sh; reviewing those for current compilers is housekeeping. |
| #159 Some CkCallback types are not valid across checkpoint/restart | 2013 | Bug | F |  | Resolved by documentation: in 2020 Evan unscheduled it because ASLR workarounds are in place and the manual says which callback types survive restart. Function-pointer callbacks across C/R are disallowed by policy (2019 core meeting). |
| #165 object location services: separate location caching from msg delivery and buffering | 2013 | cleanup | F |  | Location cache is now a separate group CkLocCache (cklocation.h:345), split out by 008ce52c5 (CkLocMgr refactor PR #3266, 2021). CkLocRec still exists but the requested separation is done. |
| #172 objid_t: Exploit compile-time declared size bounds | 2013 | feature | I |  | Compile-time index bounds: only runtime FixedArrayIndexCompressor::make(bounds) exists (cklocation.C:2476). |
| #173 objid_t: Support user-defined conversion functions | 2013 | feature | I |  | ck::ArrayIndexCompressor interface exists (ckarrayindex.h:357) but users cannot plug in their own; feature request. |
| #176 objid_t: tracing infrastructure should use objid_t  | 2013 | feature,tracing | I |  | Tracing still derives IDs from CkArrayIndex::getProjectionID(); switching to 64-bit IDs is a design improvement. |
| #177 objid_t: load balancing infrastructure should use objid_t  | 2013 | feature,load balancing | I |  | LB partially converted to 64-bit ID + omid (fde78a0); reopened in 2019 only to ask "can we do better". No defect. |
| #178 objid_t: fault tolerance infrastructure should use objid_t  | 2013 | feature,fault tolerance | O |  | Targets the message-logging FT protocols, which are a tombstone. |
| #179 objid_t: adapt callbacks to use objid_t | 2013 | feature | I |  | Callbacks still carry CkArrayIndex; moving them to 64-bit IDs is optional. |
| #180 object location services: improve implementation and protocol | 2013 | feature | I |  | Umbrella for location-service redesign; subsumed by the queued location-manager overhaul. |
| #186 ckout and ckerr should derive from std::ostream | 2013 | Bug | I |  | ckout/ckerr are still homegrown _CkOStream (ckstream.h:15), not std::ostream. API nicety. |
| #193 objid_t: Support new message type for CharmDebug | 2013 | feature | O |  | ForIDedObjMsg is only an enum value (charm.h:366); no path produces it. CharmDebug also depends on CCS, which is not ported. |
| #200 Contribute a PUP::able entity | 2013 | feature | I |  | No API to contribute a PUP::able object to a reduction; users serialize by hand into a custom reducer. |
| #235 CkCallback to array sections is very non-migratable | 2013 | Bug | K |  | Same defect as #13: bcastSection callbacks hold local pointers and CkSectionInfo, so they are valid only on the creating PE (ckcallback.C:431-437, containsPointer). |
| #246 Clean deallocation and shutdown of Charm++ processes | 2013 | Bug,mpi interoperability,shrink-expand | D |  | Original motivation (in-process restart for shrink-expand) is moot, because shrink/expand restarts via checkpoint. Still relevant only if MPI interop / CharmLibExit re-init gets ported to reconverse. |
| #253 Heterogeneous PPN support | 2013 | feature | I |  | Reconverse assumes a uniform PEs-per-process count (convcore.cpp:546 CmiNodeOf = pe / Cmi_mynodesize); heterogeneous PPN would be a new feature. |
| #259 Bugs exposed by use of randomized Q | 2013 | Bug | O |  | Randomized-queue tracking bug. CMake still accepts RANDOMIZED_MSGQ (CMakeLists.txt:183), but reconverse queueing.cpp/scheduler.cpp have no randomization, so the flag does nothing. Re-implementing it in reconverse would be a useful debugging aid (wishlist). |
| #275 Performance testing and tuning of CkIO | 2013 | feature | I |  | CkIO performance tuning (Lustre stripe hints, O_DIRECT): optional. |
| #279 Windows: Display abort message in failure pop-up | 2013 | feature | O |  | Windows: reconverse arch dirs are darwin-arm8, linux-arm8 and linux-x86_64 only. |
| #283 TLS-based AMPI variable privatization source-to-source translation tool | 2013 | feature,AMPI | A |  | AMPI TLS/ROSE privatization tool: keep for the AMPI port list. |
| #295 Non-blocking checkpointing | 2013 | feature | I |  | Non-blocking checkpointing: branch never posted; feature. |
| #297 Interoperation in VERBS layer | 2013 | feature,mpi interoperability | O |  | MPI interop in the verbs layer: classic machine layer. |
| #298 Avoid warning in sockRoutines | 2013 | Bug | O |  | sockRoutines.C is not built on reconverse (converse.cmake:142-157); the pxshm sem_open warning is classic-only. |
| #301 Interoperation in NETLRTS  layer | 2013 | feature,mpi interoperability | O |  | MPI interop in netlrts: classic machine layer. |
| #315 Reduction Starting messages - Performance optimal solution | 2013 | feature,smp | I |  | d13a2a247 (2013) removed ReductionStarting for populated PEs. PEs with no elements still receive ReductionStarting (ckreduction.C sendReductionStartingToKids). Optimization for sparse or non-migrating arrays. |
| #338 Remove "custom packed messages" in favor of PUP framework | 2013 | cleanup | I |  | Custom packed messages still exist alongside PUP; removing them is an API-cleanup decision. |
| #387 Pack variable-envelope structures to reduce overhead bytes | 2013 | feature | I |  | Envelope layout: the 2018 reorder brought it to 64 bytes; #pragma pack elsewhere is optional. |
| #424 CkIO: Automatic re-open of files across checkpoint/restart | 2014 | feature | I |  | CkIO reopening open files across checkpoint/restart: feature. |
| #437 Callbacks to array elements that should be demand-created | 2014 | feature | I |  | Callbacks to array elements use ifNotThere buffer, so they cannot demand-create elements. Hypothetical use case. |
| #443 pamilrts: Port L2 atomics support from pami layer | 2014 | feature | O |  | pamilrts L2 atomics: classic machine layer. |
| #472 Licensing of CRC32 code | 2014 | cleanup | K |  | src/util/crc32.C (unclear provenance) is still shipped and linked (ck.C:2756 record/replay checksums); with the Apache-2.0 relicense, provenance matters. A zlib-licensed replacement is identified in the issue. |
| #473 Licensing of library code in Data Transfer library | 2014 | cleanup | F |  | The datatransfer library was deleted with FEM in ad7a121d8 (#3526, 2021). |
| #474 Licensing of TreeMatchLB and supporting code | 2014 | cleanup | F |  | TreeMatchLB was removed in 33be38d6f ([lbMerge - 1] #2527, 2019). |
| #475 Licensing of parallel random number generator from NCSA | 2014 | cleanup | F |  | Reconverse replaces the NCSA Crn* generator with std::minstd_rand (contrib/reconverse src/random.cpp). The classic src/conv-core/random.C is not built and is due for deletion. |
| #480 Calling 'migrateMe' in SDAG serial block accesses members of deallocated chare | 2014 | Bug,migration | K |  | migrateMe() inside an SDAG serial block still frees the object before the SDAG continuation runs (use-after-free). manual.rst:3093 says "last action in an entry method" but does not mention SDAG. Needs a doc warning, or migrateMe should defer the migration. |
| #512 Remove duplication of conv-mach files | 2014 | cleanup | O |  | Duplicated conv-mach files live in classic src/arch/* dirs that are due for deletion; reconverse arch dirs hold only conv-mach.h/.sh. |
| #527 smp layer based on OpenMP 4.0 teams | 2014 | feature,machine layers | O |  | OpenMP-teams SMP machine layer: classic machine-layer idea. |
| #535 Errors reported by ThreadSanitizer | 2014 | cleanup,smp | O |  | ThreadSanitizer tracking bug for classic SMP (machine-smp.c etc.). A fresh TSan sweep of reconverse would be a new item (conv-mach-tsan still ships). |
| #540 CmiDestroyLocks destroys a lock that the thread (or another) is holding | 2014 | cleanup | O |  | CmiDestroyLocks in machine-smp.c machine_exit: classic conv-core. |
| #551 CkIO processor placement options are ignored | 2014 | Bug | K |  | CkIO Options activePEs/basePE/skipPEs are documented (doc/libraries/manual.rst:861-864) but ignored: procNum() returns 0 and the placement code is under #if 0 (ckio.C:558-565). trquinn reconfirmed in 2022. |
| #558 Eliminate need for code generation from .ci file with full compile-time error checking | 2014 | feature | I |  | Replace charmxi with C++ templates: long-term research direction. |
| #559 Generic (un)marshalling code to replace code generation per entry-method | 2014 | feature | I |  | Generic variadic (un)marshalling instead of per-entry generated code: long-term. |
| #560 Avoid duplicated work between CmiNodeOf and CmiRankOf | 2014 | cleanup,machine layers | O |  | On reconverse, CmiNodeOf/CmiRankOf are a single divide/modulo (convcore.cpp:546-548); the heterogeneous-PPN motivation was classic-only. |
| #571 pxshm shared queue lockless implementation is invalid | 2014 | Bug,smp | O |  | pxshm lockless queue: classic machine-layer code. |
| #595 Cut long-running tests & examples out of 'make test' | 2014 | cleanup,Build & test automation | I |  | Test-time budget still applies to the reviewed line's test tiers, but the data in the issue comes from classic CI. Housekeeping. |
| #634 Mechanism (fence/barrier) to delay execution of long methods until latency-sensitive methods are done | 2014 | feature | I |  | Fence to hold long methods until latency-sensitive ones finish: feature with several application use cases. |
| #639 method to distribute message receives across pes in node | 2014 | feature | I |  | Distribute message receives across a node's PEs: feature (the taskq work may cover it). |
| #641 protect load balancer from variable cpu clock | 2014 | feature | F |  | Root cause was comm-thread lock holding in the classic layers, fixed by 64fb65c3 and 81f00d78; a 2018 comment asked to close. |
| #650 Consolidate duplicated Fortran compiler detection/configuration | 2015 | cleanup | I |  | Duplicated Fortran detection now mostly in src/arch/common/cc-*.sh; shrinks further after classic arch deletion. Cleanup. |
| #659 Add PME functionality to LeanMD | 2015 | feature | I |  | Add PME to the LeanMD example: optional. |
| #664 charm++/communication_overhead test fails with randomized queues | 2015 | Bug | D |  | Suspected race inside benchmarks/charm++/communication_overhead (operationFinished vs next-cycle message), never confirmed. Cannot be rerun as stated, because reconverse has no randomized queue. |
| #665 charm++/delegation/multicast test fails with randomized queues due to a bug in support for anytime migration | 2015 | Bug | K |  | Duplicate of #120: section multicast + contribute + migrate hangs. Fold into #120. |
| #666 charm++/queue test fails with randomized queues | 2015 | Bug | O |  | tests/charm++/queue tests classic Cqs and is excluded on reconverse (tests/charm++/Makefile:44-50). Also not reproducible since 2015. |
| #667 charm++/sdag/migration test fails with randomized queues | 2015 | Bug | D |  | sdag/migration hang under out-of-order broadcasts. anytime_bcastred (#3947) now stress-tests migration + broadcasts on the reviewed line, but not with reordered delivery. Needs a rerun under reordering to decide. |
| #690 Broadcast to [sync] entry method should work as expected (e.g. act like passing CkCallbackResumeThread) | 2015 | feature | I |  | Broadcast to [sync] entry method acting as CkCallbackResumeThread: API feature. |
| #697 communication thread should send messages in priority order | 2015 | feature | O |  | Comm-thread send prioritization: reconverse has no comm thread. |
| #812 Merge Accel Branch | 2015 | feature | O |  | Merge Accel branch: never merged; dead feature. |
| #815 Makefile for hybrid API is not using the system OPTS | 2015 | Bug,GPU support | F |  | HAPI is now compiled by CMake (CMakeLists.txt:765/817 add_library(hybridapi)); the legacy Makefile in src/arch/cuda/hybridAPI is not used. |
| #858 improve efficiency of exclusive entry methods | 2015 | feature | K |  | Exclusive entry methods still busy-resend while the node lock is held, copying marshalled messages each time (xi-Entry.C:2187-2198, commented "RESEND CODE UNTESTED"). Wastes CPU under contention and can livelock; 16a0ba8b fixed only the no-argument free. |
| #870 SDAG methods marked as [sync] should only return when run to completion | 2015 | feature,charmxi | K |  | A [sync] entry method implemented with SDAG returns to the caller once the first SDAG block finishes: CkSendToFutureID goes in postCall right after the direct call (xi-Entry.C:2160-2186); there is no SDAG-aware completion. The caller resumes early and nothing reports it. |
| #871 Return data from [sync] SDAG methods | 2015 | feature | I |  | Return values from [sync] SDAG methods: depends on #870. |
| #881 Automatically determine location of nvcc when compiling programs using charmc in accel | 2015 | Bug,GPU support | O |  | nvcc location for accel code generation: the accel feature is dead. |
| #885 extend physical node detection across partitions | 2015 | feature | I |  | Reconverse has CmiGetPesOnPhysicalNode but no cross-partition variant; NAMD uses +devicesperreplica as a workaround. |
| #895 Refactor charmrun interface code in RTS to deduplicate netlrts/verbs | 2015 | cleanup | O |  | charmrun interface code in netlrts/verbs: charmrun is not built on reconverse. |
| #896 AMPI support for compiler-automated static variable privatization | 2015 | feature,AMPI | A |  | AMPI compiler-automated static privatization: keep for the AMPI port list. |
| #902 Projections shows garbage for the source PE of a chare array insertion event | 2015 | Bug,tracing | K |  | Projections shows a garbage "Created by PE" for dynamic-insertion events; the 2017 fix was deferred (it broke AMPI intercomm creation) and no later commit fixes it. Tracing lives on. |
| #918 multicore +commthread forces +CmiSleepOnIdle, may crash on exit | 2015 | Bug | O |  | multicore +commthread / sleep-on-idle: classic multicore layer. |
| #920 Per-GPU Node Groups | 2015 | feature,GPU support | I |  | Per-GPU nodegroups (2D nodegroups): feature request. |
| #922 Document template instantiation for registration/call wrapper code without .ci file declaration for interop use case | 2015 | feature,mpi interoperability | D |  | Documentation for interop template registration. Moot until Charm-MPI interop works on reconverse (CharmBeginInit aborts there). |

## Counts

| bucket | count |
|---|---|
| K | 11 |
| F | 8 |
| O | 27 |
| A | 2 |
| I | 42 |
| D | 7 |
| total | 97 |

## K items, ranked

No item meets the "critical" bar (a common path with a hang, crash, corruption or silent wrong result). Each K defect below sits behind a less common feature combination. The first three come closest.

1. **#870**: A `[sync]` entry method implemented in SDAG returns to its caller when the first SDAG block ends, not when the method completes. Callers resume early and nothing reports it. The code path in xi-Entry.C has not changed.
2. **#120 (+ duplicate #665)**: CkMulticast section reductions combined with anytime migration hang or assert. ckmulticast.C has had no functional fix since 2019, and the new anytime_bcastred stress test does not exercise sections.
3. **#480**: Calling `migrateMe()` inside an SDAG serial block frees the chare while the SDAG continuation still runs (use-after-free). The manual does not warn about SDAG.
4. **#13 / #235**: Section callbacks (`bcastSection`) hold raw pointers and are valid only on the creating PE. PUPing or sending one elsewhere fails. Merge the two issues.
5. **#858**: Exclusive entry methods busy-resend, and copy marshalled messages, while the node lock is held. This can livelock under contention. charmxi still labels the path "RESEND CODE UNTESTED".
6. **#551**: The CkIO placement options (activePEs, basePE, skipPEs) are documented in the libraries manual but ignored: `procNum()` returns 0. Either fix the code or remove the options from the manual.
7. **#902**: Projections records a garbage source PE for dynamic-insertion events, which breaks the timeline, the communication graph and critical-path traceback.
8. **#49**: Nothing documents or tests how dynamic insertion interacts with reductions and broadcasts. Users of insertion with reductions have to guess.
9. **#472**: src/util/crc32.C has unclear provenance, and the record/replay checksum path still links it. Now that the code is Apache-2.0, replace it with the zlib-licensed version the issue points to.

## Notes for the caller

- Several D items (#16, #246, #922) depend on one decision: whether Charm-MPI interoperation will be ported to reconverse. It is currently known broken there (CharmBeginInit aborts).
- #11 (D) depends on whether CCS returns, and #193 (O) would matter again only if CharmDebug returns with it. CCS is disabled on reconverse in cmake/detect-features.cmake.
- #259, #664, #666 and #667 all come from randomized-queue testing. The reviewed line still accepts `RANDOMIZED_MSGQ` / `--enable-randomized-msgq`, but reconverse's queue has no randomization, so the flag does nothing. Either implement it in reconverse (useful for finding ordering bugs) or remove the option.
