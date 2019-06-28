set(conv-core-c-sources cmipool.C conv-conds.C
    conv-rdma.C convcore.C cpm.C cpthreads.C cpuaffinity.C debug-conv.C futures.C
    global-nop.C isomalloc.C mem-arena.C memoryaffinity.C msgmgr.C quiescence.C
    random.C)

set(conv-core-h-sources ../util/cmitls.h cmipool.h
    conv-config.h conv-cpath.h conv-cpm.h conv-header.h conv-ooc.h
    conv-qd.h conv-random.h conv-rdma.h conv-taskQ.h conv-trace.h converse.h
    cpthreads.h debug-conv++.h debug-conv.h hrctimer.h mem-arena.h memory-gnu-threads.h
    memory-isomalloc.h msgq.h persistent.h queueing.h quiescence.h taskqueue.h)

set(conv-core-cxx-sources ../util/cmitls.C conv-interoperate.C conv-taskQ.C cputopology.C
    debug-conv++.C  memory-darwin-clang.C memory-os-wrapper.C queueing.C hrctimer.C)

set(conv-ccs-c-sources ../conv-ccs/conv-ccs.C)

set(conv-ccs-h-sources ../conv-ccs/ccs-auth.h ../conv-ccs/ccs-builtins.h
    ../conv-ccs/ccs-server.h ../conv-ccs/conv-ccs.h)

set(conv-ccs-cxx-sources ../conv-ccs/ccs-builtins.C ../conv-ccs/middle-ccs.C)

set(conv-perf-cxx-sources ../conv-perf/traceCore.C ../conv-perf/traceCoreCommon.C)

set(conv-perf-h-sources ../conv-perf/allEvents.h ../conv-perf/charmEvents.h
    ../conv-perf/charmProjections.h ../conv-perf/converseEvents.h
    ../conv-perf/converseProjections.h ../conv-perf/machineEvents.h
    ../conv-perf/machineProjections.h ../conv-perf/threadEvents.h
    ../conv-perf/traceCore.h ../conv-perf/traceCoreAPI.h
    ../conv-perf/traceCoreCommon.h)

foreach(filename ${conv-core-h-sources} ${conv-ccs-h-sources} ${conv-perf-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

add_library(ccs-client ../conv-ccs/ccs-client.C ../conv-ccs/ccs-client.h)
target_include_directories(ccs-client PRIVATE ../util)

add_library(memory-default memory.C)
target_compile_definitions(memory-default PRIVATE -DCMK_MEMORY_BUILD_DEFAULT)

add_library(memory-os memory.C)
target_compile_definitions(memory-os PRIVATE -DCMK_MEMORY_BUILD_OS)

add_library(memory-os-verbose memory.C)
target_compile_definitions(memory-os-verbose PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_VERBOSE)

add_library(memory-os-paranoid memory.C)
target_compile_definitions(memory-os-paranoid PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_PARANOID)

add_library(memory-os-leak memory.C)
target_compile_definitions(memory-os-leak PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_LEAK)

add_library(memory-os-lock memory.C)
target_compile_definitions(memory-os-lock PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_LOCK)

add_library(memory-os-isomalloc memory.C)
target_compile_definitions(memory-os-isomalloc PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_ISOMALLOC)

add_library(memory-os-wrapper memory-os-wrapper.C)

if(${CMK_HAS_MMAP})
    add_library(memory-gnu memory.C)
    target_compile_definitions(memory-gnu PRIVATE -DCMK_MEMORY_BUILD_GNU)

    add_library(memory-verbose memory.C)
    target_compile_definitions(memory-verbose PRIVATE -DCMK_MEMORY_BUILD_VERBOSE)

    add_library(memory-paranoid memory.C)
    target_compile_definitions(memory-paranoid PRIVATE -DCMK_MEMORY_BUILD_PARANOID)

    add_library(memory-leak memory.C)
    target_compile_definitions(memory-leak PRIVATE -DCMK_MEMORY_BUILD_LEAK)

    add_library(memory-isomalloc memory.C)
    target_compile_definitions(memory-isomalloc PRIVATE -DCMK_MEMORY_BUILD_ISOMALLOC)
endif()

if(${CMK_CHARMDEBUG})
    add_library(memory-os-charmdebug memory.C)
    target_compile_definitions(memory-os-charmdebug PRIVATE -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_CHARMDEBUG)

    if(${CMK_HAS_MMAP})
        add_library(memory-charmdebug memory.C)
        target_compile_definitions(memory-charmdebug PRIVATE -DCMK_MEMORY_BUILD_CHARMDEBUG)

        add_library(memory-charmdebug-slot memory.C)
        target_compile_definitions(memory-charmdebug-slot PRIVATE -DCMK_MEMORY_BUILD_CHARMDEBUG -DCMK_SEPARATE_SLOT)

        add_library(memory-charmdebug-mmap memory.C)
        target_compile_definitions(memory-charmdebug-mmap PRIVATE -DCMK_MEMORY_BUILD_CHARMDEBUG -DCPD_USE_MMAP)

        add_library(memory-charmdebug-mmap-slot memory.C)
        target_compile_definitions(memory-charmdebug-mmap-slot PRIVATE -DCMK_MEMORY_BUILD_CHARMDEBUG -DCPD_USE_MMAP -DCMK_SEPARATE_SLOT)

        add_library(memory-hooks-charmdebug memory.C)
        target_compile_definitions(memory-hooks-charmdebug PRIVATE -DCMK_MEMORY_BUILD_GNU_HOOKS -DCMK_MEMORY_BUILD_CHARMDEBUG)
    endif()
endif()


add_library(global-swap global-elfgot.C)


file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/include/commitid.c CONTENT
"extern const char * const CmiCommitID;
const char * const CmiCommitID = \"${CHARM_VERSION}\";
"
)

add_library(conv-core ${CMAKE_BINARY_DIR}/include/commitid.c ${conv-core-c-sources} ${conv-core-cxx-sources} ${conv-core-h-sources} ${conv-ccs-c-sources} ${conv-ccs-h-sources} ${conv-ccs-cxx-sources} ${conv-perf-cxx-sources} ${conv-perf-h-sources} ../conv-ldb/cldb.C)
add_dependencies(conv-core hwloc)

add_library(threads-default threads.C $<TARGET_OBJECTS:boost-context>)
target_compile_definitions(threads-default PRIVATE -DCMK_THREADS_BUILD_DEFAULT=1)
target_include_directories(threads-default PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-default-tls threads.C $<TARGET_OBJECTS:boost-context>)
target_compile_definitions(threads-default-tls PRIVATE -DCMK_THREADS_BUILD_DEFAULT=1 -DCMK_THREADS_BUILD_TLS=1)
target_include_directories(threads-default-tls PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-qt threads.C)
target_compile_definitions(threads-qt PRIVATE -DCMK_THREADS_BUILD_QT=1)
target_include_directories(threads-qt PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-qt-tls threads.C)
target_compile_definitions(threads-qt-tls PRIVATE -DCMK_THREADS_BUILD_QT=1 -DCMK_THREADS_BUILD_TLS=1)
target_include_directories(threads-qt-tls PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-context threads.C)
target_compile_definitions(threads-context PRIVATE -DCMK_THREADS_BUILD_CONTEXT=1)
target_include_directories(threads-context PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-context-tls threads.C)
target_compile_definitions(threads-context-tls PRIVATE -DCMK_THREADS_BUILD_CONTEXT=1 -DCMK_THREADS_BUILD_TLS=1)
target_include_directories(threads-context-tls PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-uJcontext threads.C)
target_compile_definitions(threads-uJcontext PRIVATE -DCMK_THREADS_BUILD_JCONTEXT=1 )
target_compile_options(threads-uJcontext PRIVATE -U_FORTIFY_SOURCE)
target_include_directories(threads-uJcontext PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-uJcontext-tls threads.C)
target_compile_definitions(threads-uJcontext-tls PRIVATE -DCMK_THREADS_BUILD_JCONTEXT=1 -DCMK_THREADS_BUILD_TLS=1)
target_compile_options(threads-uJcontext-tls PRIVATE -U_FORTIFY_SOURCE)
target_include_directories(threads-uJcontext-tls PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-uFcontext threads.C $<TARGET_OBJECTS:boost-context>)
target_compile_definitions(threads-uFcontext PRIVATE -DCMK_THREADS_BUILD_FCONTEXT=1 )
target_compile_options(threads-uFcontext PRIVATE -U_FORTIFY_SOURCE)
target_include_directories(threads-uFcontext PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util) # for qt.h uFcontext.c

add_library(threads-uFcontext-tls threads.C $<TARGET_OBJECTS:boost-context>)
target_compile_definitions(threads-uFcontext-tls PRIVATE -DCMK_THREADS_BUILD_FCONTEXT=1 -DCMK_THREADS_BUILD_TLS=1)
target_compile_options(threads-uFcontext-tls PRIVATE -U_FORTIFY_SOURCE)
target_include_directories(threads-uFcontext-tls PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util) # for qt.h uFcontext.c

add_library(threads-pthreads threads.C)
target_compile_definitions(threads-pthreads PRIVATE -DCMK_THREADS_BUILD_PTHREADS=1)
target_include_directories(threads-pthreads PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-fibers threads.C)
target_compile_definitions(threads-fibers PRIVATE -DCMK_THREADS_BUILD_FIBERS=1)
target_include_directories(threads-fibers PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-stackcopy threads.C)
target_compile_definitions(threads-stackcopy PRIVATE -DCMK_THREADS_BUILD_STACKCOPY=1)
target_include_directories(threads-stackcopy PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-memoryalias threads.C)
target_compile_definitions(threads-memoryalias PRIVATE -DCMK_THREADS_ALIAS_STACK=1)
target_include_directories(threads-memoryalias PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-qt-memoryalias threads.C)
target_compile_definitions(threads-qt-memoryalias PRIVATE -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_QT=1)
target_include_directories(threads-qt-memoryalias PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-context-memoryalias threads.C)
target_compile_definitions(threads-context-memoryalias PRIVATE -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_CONTEXT=1)
target_include_directories(threads-context-memoryalias PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)

add_library(threads-uJcontext-memoryalias threads.C)
target_compile_definitions(threads-uJcontext-memoryalias PRIVATE -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_JCONTEXT=1)
target_compile_options(threads-uJcontext-memoryalias PRIVATE -U_FORTIFY_SOURCE)
target_include_directories(threads-uJcontext-memoryalias PRIVATE ../QuickThreads ${CMAKE_BINARY_DIR}/src/QuickThreads/include ../util)


add_library(conv-static OBJECT conv-static.c)
add_custom_command(TARGET conv-core
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:conv-static> ${CMAKE_BINARY_DIR}/lib/conv-static.o
    VERBATIM
)
add_dependencies(conv-core conv-static)

add_library(conv-machine ../arch/${GDIR}/machine.C)

target_include_directories(conv-machine PRIVATE ../arch/util) # for machine*.*

target_include_directories(conv-core PRIVATE ../util) # for sockRoutines.c

add_executable(conv-cpm ../xlatcpm/conv-cpm.C)
target_compile_options(conv-cpm PRIVATE -host)
set_target_properties(conv-cpm PROPERTIES LINK_FLAGS "-host -language c++")

add_library(tsm ../langs/simplemsg/simplemsg.C ../langs/simplemsg/simplemsg.h)
add_library(sm ../langs/simplemsg/sm.C ../langs/simplemsg/sm.h)

add_library(pvmc ../langs/pvmc/pvmc_buf.C ../langs/pvmc/pvmc_comm.C ../langs/pvmc/pvmc_conv.C ../langs/pvmc/pvmc_main.C ../langs/pvmc/pvmc_pack.C ../langs/pvmc/pvmc_groups.C)

if(${F90CHARM})
    include(../ci-files.cmake)
    add_library(f90charm ../langs/f90charm/f90main.C ${CMAKE_BINARY_DIR}/include/f90main.decl.h)
    target_compile_definitions(f90charm PRIVATE -DAMPI_FORTRAN=1)
endif()
