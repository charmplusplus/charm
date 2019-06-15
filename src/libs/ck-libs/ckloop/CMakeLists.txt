set(ckloop-cxx-files CkLoop.C)
set(ckloop-h-files CkLoop.h CkLambda.h CkLoopAPI.h)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

add_library(moduleCkLoop ${ckloop-cxx-files} ${ckloop-h-files} ${CMAKE_BINARY_DIR}/include/CkLoop.decl.h)
add_dependencies(moduleCkLoop ck)
