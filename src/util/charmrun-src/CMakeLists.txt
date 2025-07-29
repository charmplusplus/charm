add_executable(charmd daemon.C daemon.h ../sockRoutines.C)
target_compile_options(charmd PRIVATE -seq -DCMK_NOT_USE_CONVERSE=1)
target_link_libraries(charmd PRIVATE -seq)

add_executable(charmd_faceless daemon.C daemon.h ../sockRoutines.C)
target_compile_options(charmd_faceless PRIVATE -seq -DFACELESS -DCMK_NOT_USE_CONVERSE=1)
target_link_libraries(charmd_faceless PRIVATE -seq)

add_executable(charmrun charmrun.C)
target_compile_options(charmrun PRIVATE -seq -DCMK_NOT_USE_CONVERSE=1)
target_link_libraries(charmrun PRIVATE -seq ckrescale)

target_include_directories(charmrun PRIVATE ../../conv-ccs ..) # for ccs-auth.c sockRoutines.c

add_dependencies(charmrun ck converse ckqt moduleNDMeshStreamer ckmain memory-default threads-default ldb-rand modulecompletion conv-static)
add_dependencies(charmd ck converse ckqt moduleNDMeshStreamer ckmain memory-default threads-default ldb-rand modulecompletion conv-static)
add_dependencies(charmd_faceless ck converse ckqt moduleNDMeshStreamer ckmain memory-default threads-default ldb-rand modulecompletion conv-static)
