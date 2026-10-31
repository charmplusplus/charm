
# conv-core
set(conv-core-h-sources
    src/util/cmitls.h
    src/conv-core/cmipool.h
    src/libs/ck-libs/tcharm/reconverse-compat/conv-config.h
    src/libs/ck-libs/tcharm/reconverse-compat/reconverse-compat.h
    src/conv-core/conv-header.h
    src/conv-core/conv-ooc.h
    src/conv-core/conv-rdmadevice.h
    src/conv-core/conv-taskQ.h
    src/conv-core/conv-trace.h
    src/conv-core/debug-conv++.h
    src/conv-core/debug-conv.h
    src/conv-core/memory-isomalloc.h
    src/conv-core/queueing.h
)

# conv-ccs
set(conv-ccs-h-sources
    src/conv-ccs/ccs-builtins.h
)

# conv-perf
set(conv-perf-cxx-sources
    src/conv-perf/traceCore.C
    src/conv-perf/traceCoreCommon.C
    src/conv-perf/machineProjections.C
)

set(conv-perf-h-sources
    src/conv-perf/allEvents.h
    src/conv-perf/charmEvents.h
    src/conv-perf/charmProjections.h
    src/conv-perf/converseEvents.h
    src/conv-perf/converseProjections.h
    src/conv-perf/machineEvents.h
    src/conv-perf/machineProjections.h
    src/conv-perf/threadEvents.h
    src/conv-perf/traceCore.h
    src/conv-perf/traceCoreAPI.h
    src/conv-perf/traceCoreCommon.h
)

# commitid
find_program(GIT git)

if(GIT AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
  execute_process(COMMAND git describe --exact-match
             OUTPUT_VARIABLE CHARM_VERSION_GIT
             RESULT_VARIABLE git_result
             OUTPUT_STRIP_TRAILING_WHITESPACE
             ERROR_QUIET
             )
  if(NOT ${git_result} EQUAL 0)
    execute_process(COMMAND git describe --long --always
             OUTPUT_VARIABLE CHARM_VERSION_GIT
             OUTPUT_STRIP_TRAILING_WHITESPACE
             )
  endif()
else()
  set(CHARM_VERSION_GIT "v${CHARM_VERSION}")
endif()

file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/include/commitid.C CONTENT
"extern const char * const CmiCommitID;
const char * const CmiCommitID = \"${CHARM_VERSION_GIT}\";
"
)

# conv-util
set(conv-util-h-sources
    src/util/cmirdmautils.h
)

# Charm++'s converse-adjacent utilities. libreconverse supplies the converse
# layer itself, the CCS core and sockRoutines; only the Charm++ CCS builtins
# (ccs-builtins.C) and the CpdList accessors behind them (debug-conv++.C) are
# built here. conv-perf and TopoManager are part of this list too.
set(conv-util-cxx-sources
    ${CMAKE_BINARY_DIR}/include/commitid.C
    src/util/cmirdmautils.C
    src/util/crc32.C
    src/util/ckdll.C
    src/util/ckhashtable.C
    src/util/ckimage.C
    src/util/conv-lists.C
    src/util/hilbert.C
    src/util/partitioning_strategies.C
    src/util/pup_c.C
    src/util/pup_cmialloc.C
    src/util/pup_paged.C
    src/util/pup_toNetwork.C
    src/util/pup_toNetwork4.C
    src/util/pup_util.C
    src/util/pup_xlater.C
    src/util/topomanager/TopoManager.C
    src/util/spanningTree.C
    ${conv-perf-cxx-sources}
    src/conv-ccs/ccs-builtins.C
    src/conv-core/debug-conv++.C
)

#Uncommenting spanning tree to satisfy ckrdma dep errors

if(CMK_CAN_LINK_FORTRAN)
    add_library(conv-utilf pup_f.f90)
    add_custom_command(TARGET conv-utilf
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/pupmod.mod ${CMAKE_BINARY_DIR}/include/
        VERBATIM
    )
endif()

add_custom_command(OUTPUT pup_f.f90 COMMAND ${CMAKE_SOURCE_DIR}/src/util/pup_f.f90.sh > /dev/null)

# conv-partition
set(conv-partition-cxx-sources
    src/util/custom_partitioner.C
    src/util/set_partition_params.C
)

# conv-ldb
set(conv-ldb-h-sources
    src/conv-ldb/topology.h
)

# TopoManager
set(tmgr-c-sources src/util/topomanager/CrayNid.c)
set(tmgr-cxx-sources src/util/topomanager/TopoManager.C)
set(tmgr-h-sources src/util/topomanager/TopoManager.h ${CMAKE_BINARY_DIR}/include/topomanager_config.h src/util/topomanager/XTTorus.h)
file(WRITE ${CMAKE_BINARY_DIR}/include/topomanager_config.h "// empty\n" )

# Converse
# The converse layer itself is libreconverse (external); charm's
# converse-adjacent utility sources build as helper libraries, and
# `converse` is an aggregation target for dependency purposes.
add_library(charm_cxx_utils STATIC
    ${conv-util-cxx-sources})
# spanningTree.C includes charm++.h -> ckmarshall.h -> CkMarshall.decl.h, a
# charmxi-generated header; without this target-level dependency the compile
# races the generation and a fresh -j build can lose (found deterministically
# on a clean tree).
add_dependencies(charm_cxx_utils ci-generated)

add_library(topomanager STATIC
    ${tmgr-cxx-sources}
    ${tmgr-h-sources}
    $<TARGET_OBJECTS:ckrescale>)
target_include_directories(topomanager PUBLIC
    src/util/topomanager
    ${CMAKE_BINARY_DIR}/include)

# Isomalloc. Reconverse has migratable user-level threads but no way to give
# one a stack at an address every process agrees on, which is what migration
# needs; that is Isomalloc's job. It cannot live inside Reconverse because
# serializing a context is written against PUP, which belongs to this layer --
# so it is built here and registered with Reconverse's thread layer through the
# CthIsomallocOps table (see isomalloc-reconverse.C). charmc links it into
# every program.
add_library(isomalloc STATIC
    src/conv-core/isomalloc.C
    src/conv-core/isomalloc-reconverse.C
    src/conv-core/memory-isomalloc.h)
add_dependencies(isomalloc reconverse charm_cxx_utils)

# Memory modules. These define malloc/free and friends for the whole program,
# so they are linked only into programs that ask for one (charmc -memory
# isomalloc, which ampicc passes); otherwise the system allocator is left
# alone. Only the Isomalloc mode is built: it is the one AMPI needs, so that a
# rank's heap allocations land in its own migratable context.
add_library(memory-os-isomalloc STATIC src/conv-core/memory.C)
target_compile_definitions(memory-os-isomalloc PRIVATE
    -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_ISOMALLOC)
add_dependencies(memory-os-isomalloc reconverse isomalloc)

# The wrapper resolves the real allocator through dlsym(RTLD_NEXT), so that a
# module that intercepts malloc still has something to fall back to.
add_library(memory-os-wrapper STATIC src/conv-core/memory-os-wrapper.C)
add_dependencies(memory-os-wrapper reconverse)

# ccs-client (the library a CCS client program links) is reconverse's
# target; it lands in lib/ with the other archives.
add_custom_target(converse)
add_dependencies(converse reconverse topomanager charm_cxx_utils ccs-client
    isomalloc memory-os-isomalloc memory-os-wrapper ckrescale)

foreach (filename
        ${conv-ldb-h-sources}
)
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)

endforeach()

foreach(filename
    ${conv-core-h-sources}
    ${conv-ccs-h-sources}
    ${conv-perf-h-sources}
    ${conv-util-h-sources}
    ${conv-ldb-h-sources}
    ${tmgr-h-sources}
)
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()
