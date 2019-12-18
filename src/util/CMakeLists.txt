set(conv-util-c-sources
    ../arch/util/lz4.c
    ../arch/util/mempool.C
    ../arch/util/persist-comm.C
    cmirdmautils.C
    crc32.C
    sockRoutines.C
)

set(conv-util-h-sources
    cmirdmautils.h
)

set(conv-util-cxx-sources
    ckdll.C
    ckhashtable.C
    ckimage.C
    conv-lists.C
    hilbert.C
    partitioning_strategies.C
    pup_c.C
    pup_cmialloc.C
    pup_paged.C
    pup_toNetwork.C
    pup_toNetwork4.C
    pup_util.C
    pup_xlater.C
    spanningTree.C
)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

configure_file(${conv-util-h-sources} ${CMAKE_BINARY_DIR}/include/ COPYONLY)

add_library(conv-util ${conv-util-c-sources} ${conv-util-cxx-sources} ${conv-util-h-sources})
add_dependencies(conv-util ck)

if(${CMK_CAN_LINK_FORTRAN})
    add_library(conv-utilf pup_f.f90)
endif()

add_custom_command(OUTPUT pup_f.f90 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/pup_f.f90.sh > /dev/null)

add_library(conv-partition custom_partitioner.C set_partition_params.C)
add_dependencies(conv-partition ck)

target_include_directories(conv-util PRIVATE . ../util/topomanager ../ck-ldb ../ck-perf ../ck-cp)
target_include_directories(conv-partition PRIVATE . ../util/topomanager ../ck-ldb ../ck-perf ../ck-cp)
