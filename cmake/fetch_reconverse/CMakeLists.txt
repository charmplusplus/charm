# This directory is solely for fetching reconverse.

set_directory_properties(PROPERTIES
  RULE_LAUNCH_COMPILE ""
  COMPILE_OPTIONS ""
  COMPILE_DEFINITIONS ""
  INCLUDE_DIRECTORIES ""
)

set(AUTOFETCH_RECONVERSE_TAG
    main
    CACHE STRING "The tag to fetch for reconverse")

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

configure_file(${reconverse_SOURCE_DIR}/include/converse.h ${reconverse_SOURCE_DIR}/include/charm-config.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
