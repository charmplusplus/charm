set(moduleNDMeshStreamer-h-sources NDMeshStreamer.h VirtualRouter.h DataItemTypes.h)

set(moduleNDMeshStreamer-cxx-sources NDMeshStreamer.C)

include(../../../ci-files.cmake)

add_library(moduleNDMeshStreamer ${moduleNDMeshStreamer-cxx-sources}
    ${moduleNDMeshStreamer-h-sources}
    ${CMAKE_BINARY_DIR}/include/NDMeshStreamer.decl.h)

target_include_directories(moduleNDMeshStreamer PUBLIC .
    ../../../util/topomanager ../../../ck-ldb ../../../ck-perf ../../../ck-cp
    ../completion)

foreach(filename ${moduleNDMeshStreamer-h-sources})
	configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()


configure_file(libmoduleNDMeshStreamer.dep ${CMAKE_BINARY_DIR}/lib/ COPYONLY)