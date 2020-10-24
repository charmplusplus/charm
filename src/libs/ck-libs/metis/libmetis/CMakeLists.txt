# This file was modified for Charm++:
# - renamed target from "metis" to "ckmetis".
# - add "file(GLOB GKlib_sources ...)"

# Add this directory for internal users.
include_directories(.)
# Find sources.
file(GLOB metis_sources *.c)
file(GLOB GKlib_sources ../GKlib/*.c)
# Build libmetis.
add_library(ckmetis ${METIS_LIBRARY_TYPE} ${GKlib_sources} ${metis_sources})
if(UNIX)
  target_link_libraries(ckmetis m)
endif()

if(METIS_INSTALL)
  install(TARGETS ckmetis
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()
