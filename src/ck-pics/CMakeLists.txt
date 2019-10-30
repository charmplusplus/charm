set(ckpics-cxx-sources picsautoperf.C picsautoperfAPI.C picsdecisiontree.C picstreenode.C trace-perf.C)

set(ckpics-h-sources picsautoperf.h picsautoperfAPI.h picsautoperfAPIC.h
	picsdecisiontree.h picsdefs.h picsdefscpp.h picstreenode.h trace-perf.h)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)
add_library(ckpics ${ckpics-cxx-sources} ${ckpics-h-sources} ${CMAKE_BINARY_DIR}/include/TraceAutoPerf.decl.h)

target_include_directories(ckpics PRIVATE . ../ck-perf ../ck-cp ../util/topomanager ../conv-ldb ../ck-ldb ../conv-perf)
