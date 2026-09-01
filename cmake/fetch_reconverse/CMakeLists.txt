# This directory is solely for fetching reconverse.

set_directory_properties(PROPERTIES
  RULE_LAUNCH_COMPILE ""
  COMPILE_OPTIONS ""
  COMPILE_DEFINITIONS ""
  INCLUDE_DIRECTORIES ""
)

# Pin an exact reconverse commit, not a branch: with a branch name, the same
# charm commit fetches whatever reconverse's tip is that day, so builds are
# unreproducible and can silently break when the pairing drifts (in Aug 2026 a
# fresh clone stopped configuring for exactly this reason). Update by bumping
# this SHA in a reviewed commit; --with-fetch-reconverse-tag/-dir still
# override for development.
set(AUTOFETCH_RECONVERSE_TAG_DEFAULT
    2c50813e71e23e5f8aec5a258c4c35e898f7144f)

set(AUTOFETCH_RECONVERSE_TAG
    ${AUTOFETCH_RECONVERSE_TAG_DEFAULT}
    CACHE STRING "The reconverse commit (full SHA) to fetch")

# A plain `set(... CACHE STRING)` writes the SHA on the first configure and then
# keeps it forever, so bumping the pin above has no effect on a build directory
# that already exists: it goes on fetching the old reconverse, and the mismatch
# surfaces as an unrelated-looking compile error rather than as "your pin did
# not take". That is exactly what the #3956 repin did locally -- the tree kept
# pre-#211 reconverse and failed on a missing CmiNodeRankOnPhysicalNode.
#
# So remember which default the cache was seeded from. If the pin in this file
# has moved and the cache still holds the previous pin, the cached value is a
# leftover rather than a decision: follow the file. A value that differs from
# both is a deliberate --with-fetch-reconverse-tag override and is left alone.
if(DEFINED _AUTOFETCH_RECONVERSE_TAG_SEED
   AND NOT _AUTOFETCH_RECONVERSE_TAG_SEED STREQUAL AUTOFETCH_RECONVERSE_TAG_DEFAULT
   AND AUTOFETCH_RECONVERSE_TAG STREQUAL _AUTOFETCH_RECONVERSE_TAG_SEED)
  message(STATUS
      "reconverse pin moved (${_AUTOFETCH_RECONVERSE_TAG_SEED} -> "
      "${AUTOFETCH_RECONVERSE_TAG_DEFAULT}); updating this build directory")
  set(AUTOFETCH_RECONVERSE_TAG ${AUTOFETCH_RECONVERSE_TAG_DEFAULT}
      CACHE STRING "The reconverse commit (full SHA) to fetch" FORCE)
endif()

set(_AUTOFETCH_RECONVERSE_TAG_SEED ${AUTOFETCH_RECONVERSE_TAG_DEFAULT}
    CACHE INTERNAL "in-tree reconverse pin this cache was seeded from")

include(FetchContent)
FetchContent_Declare(
    reconverse
    GIT_REPOSITORY https://github.com/charmplusplus/reconverse.git
    GIT_TAG ${AUTOFETCH_RECONVERSE_TAG}
)

# Set cmake variables for reconverse
set(_save_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")

FetchContent_MakeAvailable(reconverse)

set(BUILD_SHARED_LIBS ${_save_BUILD_SHARED_LIBS} CACHE INTERNAL "")

configure_file(${reconverse_SOURCE_DIR}/include/converse.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
configure_file(${reconverse_SOURCE_DIR}/src/cldb.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
configure_file(${reconverse_SOURCE_DIR}/include/charm-config.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
# Charm++ must build against the same RDMA configuration that libreconverse was
# compiled with (CMK_REG_REQUIRED, CMK_NOCOPY_DIRECT_BYTES). The legacy
# src/conv-core/conv-rdma.h is excluded from the Converse header copy in
# cmake/converse.cmake so that this is the only conv-rdma.h on the include path.
configure_file(${reconverse_SOURCE_DIR}/include/conv-rdma.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
# reconverse's converse.h includes persistent.h, so this must be the copy on
# the include path. The legacy src/conv-core/persistent.h is excluded from the
# Converse header copy in cmake/converse.cmake, and its machine-layer
# implementation (src/arch/util/persist-comm.C) is dropped from the build
# there as well.
configure_file(${reconverse_SOURCE_DIR}/include/persistent.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
# Likewise taskqueue.h, the other header reconverse's converse.h includes. Note
# that this copy does not include converse.h itself, so anything including it
# directly has to include converse.h first.
configure_file(${reconverse_SOURCE_DIR}/include/taskqueue.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
