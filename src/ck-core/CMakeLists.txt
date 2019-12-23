set(ck-cxx-sources
    ../ck-cp/API-generator/cp_effects.C ../ck-cp/arrayRedistributor.C
    ../ck-cp/controlPoints.C ../ck-cp/pathHistory.C
    ../ck-ldb/BaseLB.C ../ck-ldb/CentralLB.C
    ../ck-ldb/CentralPredictor.C ../ck-ldb/ckgraph.C
    ../ck-ldb/ckheap.C ../ck-ldb/ckset.C
    ../ck-ldb/DistBaseLB.C ../ck-ldb/forestTest.C
    ../ck-ldb/HybridBaseLB.C
    ../ck-ldb/LBComm.C ../ck-ldb/LBDatabase.C
    ../ck-ldb/lbdbf.C
    ../ck-ldb/LBMachineUtil.C
    ../ck-ldb/LBObj.C
    ../ck-ldb/LBSimulation.C ../ck-ldb/LButil.C
    ../ck-ldb/manager.C ../ck-ldb/MetaBalancer.C
    ../ck-ldb/NullLB.C ../ck-ldb/readmodel.C
    ../ck-ldb/Refiner.C ../ck-ldb/RefinerTemp.C
    ../ck-ldb/treeTest.C ../ck-ldb/weakTest.C
    ../ck-perf/trace-common.C ../ck-perf/trace-controlPoints.C
    ../ck-perf/tracec.C ../ck-perf/tracef.C ../conv-perf/charmProjections.C
    ../util/ckbitvector.C ../util/ckregex.C
    ck.C ckarray.C ckarrayoptions.C ckcallback.C
    ckcheckpoint.C ckevacuation.C ckfutures.C ckIgetControl.C
    cklocation.C ckmemcheckpoint.C ckmulticast.C
    ckobjid.C ckobjQ.C ckrdma.C ckreduction.C cktiming.C debug-charm.C
    debug-message.C init.C modifyScheduler.C mpi-interoperate.C msgalloc.C
    qd.C register.C sdag.C waitqd.C
)

set(ck-h-sources XArraySectionReducer.h charm++.h charm++_type_traits.h
    charm-api.h charm.h charmf.h ck.h ckIgetControl.h ckarray.h ckarrayindex.h
    ckarrayoptions.h ckcallback-ccs.h ckcallback.h ckcheckpoint.h
    ckevacuation.h ckfutures.h cklocation.h cklocrec.h
    ckmemcheckpoint.h ckmessage.h ckmigratable.h ckmulticast.h
    ckobjQ.h ckobjid.h ckrdma.h ckreduction.h cksection.h
    ckstream.h cktiming.h debug-charm.h envelope-path.h envelope.h init.h
    middle-blue.h middle-conv.h middle.h mpi-interoperate.h objid.h qd.h
    readonly.h register.h sdag.h stats.h waitqd.h ../ck-ldb/CentralLB.h
    ../ck-cp/arrayRedistributor.h ../ck-cp/controlPoints.h ../ck-cp/controlPointsf.h ../ck-cp/pathHistory.h ../ck-cp/API-generator/cp_effects.h)

set(ck-ci-sources
    ${CMAKE_BINARY_DIR}/include/BaseLB.decl.h
    ${CMAKE_BINARY_DIR}/include/CentralLB.decl.h
    ${CMAKE_BINARY_DIR}/include/CkArray.decl.h
    ${CMAKE_BINARY_DIR}/include/CkCallback.decl.h
    ${CMAKE_BINARY_DIR}/include/CkCheckpoint.decl.h
    ${CMAKE_BINARY_DIR}/include/CkCheckpointStatus.decl.h
    ${CMAKE_BINARY_DIR}/include/CkFutures.decl.h
    ${CMAKE_BINARY_DIR}/include/CkLocation.decl.h
    ${CMAKE_BINARY_DIR}/include/CkMarshall.decl.h
    ${CMAKE_BINARY_DIR}/include/CkMemCheckpoint.decl.h
    ${CMAKE_BINARY_DIR}/include/CkMulticast.decl.h
    ${CMAKE_BINARY_DIR}/include/CkReduction.decl.h
    ${CMAKE_BINARY_DIR}/include/ControlPoints.decl.h
    ${CMAKE_BINARY_DIR}/include/completion.decl.h
    ${CMAKE_BINARY_DIR}/include/DistBaseLB.decl.h
    ${CMAKE_BINARY_DIR}/include/DistributedLB.decl.h
    ${CMAKE_BINARY_DIR}/include/f90main.decl.h
    ${CMAKE_BINARY_DIR}/include/GreedyLB.decl.h
    ${CMAKE_BINARY_DIR}/include/HybridBaseLB.decl.h
    ${CMAKE_BINARY_DIR}/include/LBDatabase.decl.h
    ${CMAKE_BINARY_DIR}/include/MetaBalancer.decl.h
    ${CMAKE_BINARY_DIR}/include/mpi_main.decl.h
    ${CMAKE_BINARY_DIR}/include/NullLB.decl.h
    ${CMAKE_BINARY_DIR}/include/PathHistory.decl.h
    ${CMAKE_BINARY_DIR}/include/RefineLB.decl.h
    ${CMAKE_BINARY_DIR}/include/TraceControlPoints.decl.h
    ${CMAKE_BINARY_DIR}/include/waitqd.decl.h
)

set(ldb-cxx-sources  ../ck-ldb/BlockLB.C
    ../ck-ldb/CommonLBs.C  ../ck-ldb/DistributedLB.C
    ../ck-ldb/DummyLB.C ../ck-ldb/EveryLB.C
    ../ck-ldb/GreedyLB.C ../ck-ldb/GreedyRefineLB.C
    ../ck-ldb/LButil.C
    ../ck-ldb/RandCentLB.C ../ck-ldb/RefineLB.C
    ../ck-ldb/Refiner.C ../ck-ldb/RefinerTemp.C
    ../ck-ldb/RotateLB.C ../ck-ldb/TempAwareRefineLB.C
    ../ck-ldb/ckgraph.C ../ck-ldb/ckheap.C
    ../ck-ldb/ckset.C  ../ck-ldb/manager.C)


set(ldb-h-sources ../ck-ldb/BaseLB.h ../ck-ldb/CentralLBMsg.h
    ../ck-ldb/DistBaseLB.h ../ck-ldb/DistributedLB.h ../ck-ldb/DummyLB.h
    ../ck-ldb/GreedyLB.h ../ck-ldb/GreedyRefineLB.h
    ../ck-ldb/HybridBaseLB.h ../ck-ldb/HybridLBMsg.h
    ../ck-ldb/LBComm.h
    ../ck-ldb/LBDatabase.h ../ck-ldb/LBMachineUtil.h ../ck-ldb/LBOM.h
    ../ck-ldb/LBObj.h ../ck-ldb/LBSimulation.h
    ../ck-ldb/MetaBalancer.h ../ck-ldb/MetisLB.h
    ../ck-ldb/NullLB.h
    ../ck-ldb/RandCentLB.h ../ck-ldb/RandomForestModel.h
    ../ck-ldb/RefineLB.h ../ck-ldb/Refiner.h
    ../ck-ldb/RefinerTemp.h ../ck-ldb/RotateLB.h
    ../ck-ldb/ScotchLB.h ../ck-ldb/ScotchRefineLB.h
    ../ck-ldb/ScotchTopoLB.h ../ck-ldb/TempAwareRefineLB.h
    ../ck-ldb/ZoltanLB.h ../ck-ldb/ckgraph.h
    ../ck-ldb/ckheap.h ../ck-ldb/ckset.h
    ../ck-ldb/elements.h
    ../ck-ldb/lbdb.h ../ck-ldb/manager.h)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

add_library(ck ${ck-cxx-sources} ${ck-h-sources} ${ck-ci-sources} ${all-ci-outputs})
add_library(ckmain main.C)
add_dependencies(ckmain ck)

add_library(moduleCommonLBs
    ../ck-ldb/CommonLBs.C
    ../ck-ldb/DummyLB.C ../ck-ldb/GreedyLB.C ../ck-ldb/GreedyRefineLB.C
    ../ck-ldb/RandCentLB.C ../ck-ldb/RefineLB.C
    ../ck-ldb/RotateLB.C ../ck-ldb/DistributedLB.C
    ../ck-ldb/manager.C
    ${CMAKE_BINARY_DIR}/include/CommonLBs.decl.h
    ${CMAKE_BINARY_DIR}/include/DummyLB.decl.h
    ${CMAKE_BINARY_DIR}/include/GreedyRefineLB.decl.h
    ${CMAKE_BINARY_DIR}/include/RandCentLB.decl.h
    ${CMAKE_BINARY_DIR}/include/RotateLB.decl.h
    ${CMAKE_BINARY_DIR}/include/MetisLB.decl.h
    ${ck-ci-sources}
)

if(${EVERYLB})
add_library(moduleEveryLB
    ${ldb-cxx-sources}
    ${ldb-h-sources}
    ${ck-ci-sources}
    ${CMAKE_BINARY_DIR}/include/CommonLBs.decl.h
    ${CMAKE_BINARY_DIR}/include/EveryLB.decl.h
    ${CMAKE_BINARY_DIR}/include/TempAwareRefineLB.decl.h
    ${CMAKE_BINARY_DIR}/include/DummyLB.decl.h
    ${CMAKE_BINARY_DIR}/include/GreedyRefineLB.decl.h
    ${CMAKE_BINARY_DIR}/include/RotateLB.decl.h
    ${CMAKE_BINARY_DIR}/include/MetisLB.decl.h
)
endif()

# Single LBs
add_library(moduleDistributedLB ../ck-ldb/DistributedLB.C ${CMAKE_BINARY_DIR}/include/DistributedLB.decl.h ${ck-ci-sources})
add_library(moduleDummyLB ../ck-ldb/DummyLB.C ${CMAKE_BINARY_DIR}/include/DummyLB.decl.h ${ck-ci-sources})
add_library(moduleGreedyLB ../ck-ldb/GreedyLB.C ${CMAKE_BINARY_DIR}/include/GreedyLB.decl.h ${ck-ci-sources})
add_library(moduleGreedyRefineLB ../ck-ldb/GreedyRefineLB.C ${CMAKE_BINARY_DIR}/include/GreedyRefineLB.decl.h ${ck-ci-sources})
add_library(moduleRandCentLB ../ck-ldb/RandCentLB.C ${CMAKE_BINARY_DIR}/include/RandCentLB.decl.h ${ck-ci-sources})
add_library(moduleRefineLB ../ck-ldb/RefineLB.C ${CMAKE_BINARY_DIR}/include/RefineLB.decl.h ${ck-ci-sources})
add_library(moduleRotateLB ../ck-ldb/RotateLB.C ${CMAKE_BINARY_DIR}/include/RotateLB.decl.h ${CMAKE_BINARY_DIR}/include/CentralLB.decl.h ${ck-ci-sources})

add_library(moduleCkMulticast ../libs/ck-libs/dummy/ckdummy.C)

add_library(moduleCkIO ../libs/ck-libs/io/ckio.C ../libs/ck-libs/io/fs_parameters.c ../libs/ck-libs/io/ckio.h ${CMAKE_BINARY_DIR}/include/CkIO.decl.h)
add_dependencies(moduleCkIO ck)
configure_file(../libs/ck-libs/io/ckio.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)

foreach(filename ${ck-h-sources} ${ldb-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

target_include_directories(ck PRIVATE . ../ck-ldb ../ck-perf ../ck-cp ../util/topomanager ../conv-perf)
target_include_directories(ckmain PRIVATE . ../ck-ldb ../ck-perf ../ck-cp ../util/topomanager)

# Fortran interface
if(${CMK_CAN_LINK_FORTRAN})
    add_library(ckf charmf.C charmmod.f90 charmf.h ${ck-ci-sources})
    add_library(ckmainf mainf.f90)
endif()

add_library(mpi-mainmodule mpi-mainmodule.C ${ck-ci-sources})
