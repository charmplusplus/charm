set(ckperf-h-sources trace-counter.h trace-common.h
    trace-memory.h trace-projector.h
    trace.h tracec.h)

foreach(filename ${ckperf-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

if(CMK_TRACE_ENABLED)
    add_library(trace-projector trace-projector.C ../conv-perf/traceCore.C
        ../conv-perf/traceCoreCommon.C ../conv-perf/charmProjections.C
        ../conv-perf/converseProjections.C ../conv-perf/machineProjections.C)
    add_dependencies(trace-projector ck)

    add_library(trace-counter trace-counter.C)
    add_dependencies(trace-counter ck)

    add_library(trace-memory trace-memory.C)
    add_dependencies(trace-memory ck)
endif()

add_library(trace-converse trace-converse.C)

if(CMK_CAN_LINK_FORTRAN)
    add_library(tracef_f tracef_f.f90)
    add_custom_command(TARGET tracef_f
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/tracemod.mod ${CMAKE_BINARY_DIR}/include/
        VERBATIM
    )
endif()
