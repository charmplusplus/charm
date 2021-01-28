set(ckpics-cxx-sources picsautoperf.C picsautoperfAPI.C picsdecisiontree.C picstreenode.C trace-perf.C)

set(ckpics-h-sources picsautoperf.h picsautoperfAPI.h picsautoperfAPIC.h
	picsdecisiontree.h picsdefs.h picsdefscpp.h picstreenode.h trace-perf.h)

add_library(trace-perfReport ${ckpics-cxx-sources} ${ckpics-h-sources})

add_dependencies(trace-perfReport ck)

target_include_directories(trace-perfReport PRIVATE . ../ck-perf ../ck-cp ../util/topomanager ../conv-ldb ../ck-ldb ../conv-perf)
