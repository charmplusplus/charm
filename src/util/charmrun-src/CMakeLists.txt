add_executable(charmd daemon.C daemon.h ../sockRoutines.C)
target_compile_options(charmd PRIVATE -seq -DCMK_NOT_USE_CONVERSE=1)

add_executable(charmd_faceless daemon.C daemon.h ../sockRoutines.C)
target_compile_options(charmd_faceless PRIVATE -seq -DFACELESS -DCMK_NOT_USE_CONVERSE=1)

add_executable(charmrun charmrun.C)
target_compile_options(charmrun PRIVATE -seq -DCMK_NOT_USE_CONVERSE=1)

target_include_directories(charmrun PRIVATE ../../conv-ccs ..) # for ccs-auth.c sockRoutines.c

add_dependencies(charmrun conv-util ck conv-core ckqt moduleNDMeshStreamer ckmain memory-default threads-default conv-machine tmgr conv-util conv-partition hwloc ldb-rand conv-ldb modulecompletion conv-static)
add_dependencies(charmd conv-util ck conv-core ckqt moduleNDMeshStreamer ckmain memory-default threads-default conv-machine tmgr conv-util conv-partition hwloc ldb-rand conv-ldb modulecompletion conv-static)
add_dependencies(charmd_faceless conv-util ck conv-core ckqt moduleNDMeshStreamer ckmain memory-default threads-default conv-machine tmgr conv-util conv-partition hwloc ldb-rand conv-ldb modulecompletion conv-static)
