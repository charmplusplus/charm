set(ckperf-h-sources trace-Tau.h trace-TauBOC.h trace-bluegene.h
    trace-controlPoints.h trace-controlPointsBOC.h trace-counter.h trace-common.h
    trace-memory.h trace-projections.h trace-projectionsBOC.h trace-projector.h
    trace-simple.h trace-simpleBOC.h trace-summary.h trace-summaryBOC.h
    trace-utilization.h trace.h tracec.h)

foreach(filename ${ckperf-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

if($CMK_TRACE_ENABLED})
    add_library(trace-projector trace-projector.C ../conv-perf/traceCore.C
        ../conv-perf/traceCoreCommon.C ../conv-perf/charmProjections.C
        ../conv-perf/converseProjections.C ../conv-perf/machineProjections.C)
    add_library(trace-projections trace-projections.C ${CMAKE_BINARY_DIR}/include/TraceProjections.decl.h)
    add_library(trace-controlPoints trace-controlPoints.C)
    add_library(trace-perfReport ../ck-pics/trace-perf.C ../ck-pics/picsautoperf.C ../ck-pics/picsautoperfAPI.C ../ck-pics/picstreenode.C ../ck-pics/picsdecisiontree.C ${CMAKE_BINARY_DIR}/include/TraceAutoPerf.decl.h)
    add_library(trace-summary trace-summary.C ${CMAKE_BINARY_DIR}/include/TraceSummary.decl.h)
    add_library(trace-utilization trace-utilization.C ${CMAKE_BINARY_DIR}/include/TraceUtilization.decl.h)
    add_library(trace-simple trace-simple.C ${CMAKE_BINARY_DIR}/include/TraceSimple.decl.h)
    add_library(trace-counter trace-counter.C)
    # FIXME: add bluegene
    # add_library(trace-bluegene trace-bluegene.C)
    add_library(trace-memory trace-memory.C)
endif()
add_library(trace-converse trace-converse.C)

if(${CMK_CAN_LINK_FORTRAN})
    add_library(tracef_f tracef_f.f90)
endif()
