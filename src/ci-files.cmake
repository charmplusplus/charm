set(ci-files
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckarray.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckcallback.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckcheckpoint.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckcheckpointstatus.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckfutures.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/cklocation.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckmarshall.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckmemcheckpoint.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckmulticast.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/ckreduction.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/mpi-mainmodule.ci
    ${CMAKE_SOURCE_DIR}/src/ck-core/waitqd.ci
    ${CMAKE_SOURCE_DIR}/src/ck-cp/controlPoints.ci
    ${CMAKE_SOURCE_DIR}/src/ck-cp/controlPointsNoTrace.ci
    ${CMAKE_SOURCE_DIR}/src/ck-cp/pathHistory.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/AdaptiveLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/BaseLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/BlockLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/CentralLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/ComboCentLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/CommAwareRefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/CommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/CommonLBs.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/DistBaseLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/DistributedLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/DummyLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/EveryLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GraphBFTLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GraphPartLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GreedyAgentLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GreedyCommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GreedyLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GreedyRefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GridCommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GridCommRefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GridHybridLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GridHybridSeedLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/GridMetisLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/HbmLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/HierarchicalLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/HybridBaseLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/HybridLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/LBDatabase.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/MetaBalancer.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/MetisLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/NborBaseLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/NeighborCommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/NeighborLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/NodeLevelLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/NullLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/OrbLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/PhasebyArrayLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RandCentLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RecBipartLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RecBisectBfLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RefineCommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RefineKLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RefineSwapLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RefineTopoLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/RotateLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/ScotchLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/ScotchRefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/ScotchTopoLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TeamLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TempAwareCommLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TempAwareGreedyLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TempAwareRefineLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TopoCentLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TopoLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/TreeMatchLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/WSLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-ldb/ZoltanLB.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-controlPoints.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-projections.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-simple.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-summary.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-Tau.ci
    ${CMAKE_SOURCE_DIR}/src/ck-perf/trace-utilization.ci
    ${CMAKE_SOURCE_DIR}/src/ck-pics/picsautoperf.ci
    ${CMAKE_SOURCE_DIR}/src/langs/bluegene/BlueGene.ci
    ${CMAKE_SOURCE_DIR}/src/langs/charj/tests/jacobi/hand_translation/jacobi.ci
    ${CMAKE_SOURCE_DIR}/src/langs/charj/tests/jacobi/reference/parallelJacobi.ci
    ${CMAKE_SOURCE_DIR}/src/langs/f90charm/f90main.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ampi/ampi.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/amr/amr.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/armci/armci.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/barrier/barrier.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/cache/CkCache.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ckloop/CkLoop.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/collide/collidecharm.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/collide/threadCollide.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/completion/completion.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/dummy/ckdummy.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/fem/fem.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/fem/fem_mesh_modify.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/fftlib/fftlib.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/io/ckio.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/irecv/receiver.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/liveViz/liveViz.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/liveViz/liveVizPoll.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/liveViz3d/liveViz3d.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/liveViz3d/lv3d0.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/liveViz3d/lv3d1.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/mblock/mblock.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/MeshStreamer/MeshStreamer.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/multiphaseSharedArrays/msa-DistPageMgr.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/NDMeshStreamer/NDMeshStreamer.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/netfem/netfem.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ParFUM-Iterators/ParFUM_Iterators.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ParFUM-Tops/ParFUM_TOPS.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ParFUM/mesh_modify.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ParFUM/ParFUM.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/ParFUM/ParFUM_SA.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pencilfft/pencilfft.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pmaf/chunk.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pmaf/pgm.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/evmpool.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/gvt.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/ldbal.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/memory_temporal.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/mempool.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/pose.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/sim.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pose/stats.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pythonCCS/charmdebug-python.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/pythonCCS/PythonCCS.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/search/search.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/sparseContiguousReducer/cksparsecontiguousreducer.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/sparseReducer/cksparsereducer.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/state_space_searchengine/searchEngine.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/taskGraph/taskGraph.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/tcharm/tcharm.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/tcharm/tcharmmain.ci
    ${CMAKE_SOURCE_DIR}/src/libs/ck-libs/TMRC2D/refine.ci
    ${CMAKE_SOURCE_DIR}/src/libs/conv-libs/openmp_llvm/runtime/ompcharm/OmpCharm.ci
)

foreach(in_f ${ci-files})

    # Special handling for ci files whose filename is not the same as the module name
    if(${in_f} MATCHES src/ck-pics/picsautoperf.ci)
        set(ci-output TraceAutoPerf.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/io/ckio.ci)
        set(ci-output CkIO.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckreduction.ci)
        set(ci-output CkReduction.decl.h)
    elseif(${in_f} MATCHES src/ck-core/cklocation.ci)
        set(ci-output CkLocation.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckmulticast.ci)
        set(ci-output CkMulticast.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckarray.ci)
        set(ci-output CkArray.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckmarshall.ci)
        set(ci-output CkMarshall.decl.h)
    elseif(${in_f} MATCHES src/ck-cp/pathHistory.ci)
        set(ci-output PathHistory.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckfutures.ci)
        set(ci-output CkFutures.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckmemcheckpoint.ci)
        set(ci-output CkMemCheckpoint.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckcheckpointstatus.ci)
        set(ci-output CkCheckpointStatus.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckcallback.ci)
        set(ci-output CkCallback.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckcheckpoint.ci)
        set(ci-output CkCheckpoint.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-controlPoints.ci)
        set(ci-output TraceControlPoints.decl.h)
    elseif(${in_f} MATCHES src/ck-cp/controlPoints.ci)
        set(ci-output ControlPoints.decl.h)
    elseif(${in_f} MATCHES src/ck-core/mpi-mainmodule.ci)
        set(ci-output mpi_main.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/multiphaseSharedArrays/msa-DistPageMgr.ci)
        set(ci-output msa.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/fem/fem_mesh_modify.ci)
        set(ci-output FEMMeshModify.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/pythonCCS/charmdebug-python.ci)
        set(ci-output charmdebug_python.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/ParFUM/mesh_modify.ci)
        set(ci-output ParFUM_Adapt.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/collide/threadCollide.ci)
        set(ci-output collide.decl.h)
    elseif(${in_f} MATCHES src/ck-core/ckarray.ci)
        set(ci-output CkArray.decl.h)
    # elseif(${in_f} MATCHES src/libs/ck-libs/tmr/tri.ci)
    #     set(ci-output refine.decl.h)
    # elseif(${in_f} MATCHES src/libs/ck-libs/TMRC2D/old_pgm.ci)
    #     set(ci-output Pgm.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/TMRC2D/refine.ci)
        set(ci-output TMR_Interface.decl.h)
    elseif(${in_f} MATCHES src/libs/ck-libs/pmaf/chunk.ci)
        set(ci-output PMAF.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-Tau.ci)
        set(ci-output TraceTau.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-summary.ci)
        set(ci-output TraceSummary.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-simple.ci)
        set(ci-output TraceSimple.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-projections.ci)
        set(ci-output TraceProjections.decl.h)
    elseif(${in_f} MATCHES src/ck-perf/trace-utilization.ci)
        set(ci-output TraceUtilization.decl.h)

    else()
        # ci filename equal to module name
        get_filename_component(ci-output ${in_f} NAME_WE)
        string(APPEND ci-output ".decl.h")
    endif()

    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/include/${ci-output}
      COMMAND ${CMAKE_C_COMPILER_LAUNCHER} -I. ${in_f}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/include/
      DEPENDS ${in_f} charmxi
      )
endforeach()
