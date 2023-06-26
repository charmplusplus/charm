set(ck-cxx-sources
    ../conv-perf/charmProjections.C
    ../util/ckbitvector.C ../util/ckregex.C
    ckreduction.C debug-charm.C
    debug-message.C msgalloc.C
)

set(ck-h-sources
    charm-api.h charm.h)


include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

#add_library(ck ${ck-cxx-sources} ${ck-h-sources} ${all-ci-outputs})
#add_library(ckmain main.C)
#add_dependencies(ckmain ck)

# Single LBs

foreach(filename ${ck-h-sources} ${ldb-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

#target_include_directories(ck PRIVATE . ../ck-perf ../util/topomanager ../conv-perf)
#target_include_directories(ckmain PRIVATE . ../ck-perf ../util/topomanager)

#add_library(mpi-mainmodule mpi-mainmodule.C)
#add_dependencies(mpi-mainmodule ck)
