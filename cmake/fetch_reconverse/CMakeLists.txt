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
set(AUTOFETCH_RECONVERSE_TAG
    f3f411072bc9790232c42e22782191ba83749ba6
    CACHE STRING "The reconverse commit (full SHA) to fetch")

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
