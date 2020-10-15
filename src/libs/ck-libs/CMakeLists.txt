## The targets in this file are only built when building LIBS

# cache
add_library(moduleCkCache cache/CkCache.C cache/CkCache.h)
configure_file(cache/CkCache.h ${CMAKE_BINARY_DIR}/include COPYONLY)
add_dependencies(moduleCkCache ck)

# sparseContiguousReducer
add_library(moduleCkSparseContiguousReducer sparseContiguousReducer/cksparsecontiguousreducer.h sparseContiguousReducer/cksparsecontiguousreducer.C)
add_dependencies(moduleCkSparseContiguousReducer ck)
configure_file(sparseContiguousReducer/cksparsecontiguousreducer.h ${CMAKE_BINARY_DIR}/include COPYONLY)

# multiphaseSharedArrays

set(msa-h-sources multiphaseSharedArrays/msa.h multiphaseSharedArrays/msa-common.h multiphaseSharedArrays/msa-distArray.h multiphaseSharedArrays/msa-DistPageMgr.h)
add_library(modulemsa ${msa-h-sources} multiphaseSharedArrays/msa-distArray.C)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/msa)

foreach(file ${msa-h-sources})
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/msa COPYONLY)
endforeach()

add_dependencies(modulemsa ck)

# armci
configure_file(armci/libmodulearmci.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(armci/armci.h ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(modulearmci armci/armci_vp.C armci/armci_impl.h armci/armci_api.C armci/armci.h)
add_dependencies(modulearmci ck)

# collide
set(collide-h-sources collide/collide_util.h collide/bbox.h
    collide/collide_cfg.h collide/collidecharm.h collide/collidec.h
    collide/collidef.h collide/collidecharm_impl.h collide/collide_serial.h
    collide/collide_buffers.h)
set(collide-cxx-sources collide/collide_util.C collide/collide_serial.C
collide/collidecharm.C collide/collide_buffers.C)

foreach(file collide/collide_util.h collide/bbox.h collide/collide_cfg.h
    collide/collide_buffers.h collide/collidecharm.h collide/collidec.h
    collide/collidef.h)
    configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
endforeach(file)
configure_file(collide/libmodulecollide.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)
add_library(modulecollidecharm ${collide-h-sources} ${collide-cxx-sources})
add_library(modulecollide ${collide-h-sources} ${collide-cxx-sources} collide/threadCollide.C)
add_dependencies(modulecollide ck)
add_dependencies(modulecollidecharm ck)

# mblock
configure_file(mblock/libmodulemblock.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(mblock/mblock.h  ${CMAKE_BINARY_DIR}/include COPYONLY)
configure_file(mblock/mblockf.h  ${CMAKE_BINARY_DIR}/include COPYONLY)

add_library(modulemblock mblock/mblock.C mblock/patch.C
mblock/compat_driver.c mblock/compat_fdriver.c mblock/mblock.h
mblock/mblockf.h mblock/patch.h mblock/gridutil.h mblock/mblock_impl.h)
add_dependencies(modulemblock ck)

# barrier
configure_file(barrier/barrier.h  ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(modulebarrier barrier/barrier.C barrier/barrier.h)
add_dependencies(modulebarrier ck)

# irecv
configure_file(irecv/receiver.h  ${CMAKE_BINARY_DIR}/include COPYONLY)
configure_file(irecv/receiverf.h  ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(irecv irecv/receiver.C irecv/receiver.h)
add_dependencies(irecv ck)

# liveViz
set(liveviz-h-sources liveViz/liveViz.h liveViz/liveViz0.h liveViz/colorScale.h liveViz/ImageData.h)
set(liveviz-cxx-sources liveViz/liveViz0.C liveViz/liveViz.C liveViz/liveVizPoll.C liveViz/colorScale.C liveViz/ImageData.C liveViz/compat_float2rgb.C)

add_library(moduleliveViz ${liveviz-h-sources} ${liveviz-cxx-sources})
add_dependencies(moduleliveViz ck)

foreach(file ${liveviz-h-sources})
    configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
endforeach()

target_compile_options(moduleliveViz PRIVATE -DEXTERIOR_BLACK_PIXEL_ELIMINATION)
if(CMK_USE_LIBJPEG)
    target_include_directories(moduleliveViz PRIVATE ${JPEG_INCLUDE_DIRS})
endif()

file(WRITE ${CMAKE_BINARY_DIR}/lib/libmoduleliveViz.dep "${CMK_LIBJPEG}\n")

# taskGraph
configure_file(taskGraph/taskGraph.h ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(moduletaskGraph taskGraph/taskGraph.h taskGraph/taskGraph.C)
add_dependencies(moduletaskGraph ck)


# search
set(search-h-sources search/search.h search/problem.h search/serialtree.h search/idastar.h)
set(search-cxx-sources search/search.C search/problem.C search/serialtree.C search/idastar.C)

foreach(file ${search-h-sources})
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/cklibs/ COPYONLY)
endforeach()

add_library(modulesearch ${search-h-sources} ${search-cxx-sources})
target_compile_options(modulesearch PRIVATE -DRANDOM_STARTING_PROC)
add_dependencies(modulesearch ck)

# MeshStreamer

configure_file(MeshStreamer/MeshStreamer.h ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(moduleMeshStreamer MeshStreamer/MeshStreamer.h MeshStreamer/MeshStreamer.C)
add_dependencies(moduleMeshStreamer ck)


# pose
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/pose/)

set(pose-cxx-sources pose/eventID.C pose/stats.C pose/srtable.C pose/cancel.C
pose/memory_temporal.C
pose/mempool.C pose/pvtobj.C pose/lbObject.C pose/ldbal.C pose/gvt.C pose/event.C pose/eqheap.C
pose/evq.C pose/sim.C pose/rep.C pose/strat.C pose/seq.C pose/cons.C pose/opt.C pose/opt2.C pose/opt3.C
pose/spec.C pose/adapt.C pose/adapt2.C pose/adapt3.C pose/adapt4.C pose/adapt5.C pose/pose.C)

add_library(modulepose ${pose-cxx-sources})
add_library(moduleseqpose ${pose-cxx-sources})
target_compile_options(moduleseqpose PRIVATE -DSEQUENTIAL_POSE=1)
add_dependencies(modulepose ck)
add_dependencies(moduleseqpose ck)

configure_file(pose/libmodulepose.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(pose/libmodulepose.dep ${CMAKE_BINARY_DIR}/lib/libmoduleseqpose.dep COPYONLY)


# idxl
set(idxl-h-sources idxl/idxlc.h idxl/idxlf.h idxl/idxl.h idxl/idxl_layout.h idxl/idxl_comm.h)
set(idxl-cxx-sources idxl/idxl.C idxl/idxl_layout.C idxl/idxl_comm.C idxl/idxl_api.C)
add_library(idxl ${idxl-h-sources} ${idxl-cxx-sources})
add_dependencies(idxl ck moduleampi)

set_target_properties(idxl PROPERTIES RULE_LAUNCH_COMPILE ${CMAKE_BINARY_DIR}/bin/ampicxx)

foreach(file ${idxl-h-sources})
	configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
endforeach()


# searchengine

set(searchengine-h-sources state_space_searchengine/searchEngine.h state_space_searchengine/searchEngine_impl.h)

set(searchengine-cxx-sources state_space_searchengine/searchEngine.C)

add_library(modulesearchEngine ${searchengine-h-sources} ${searchengine-cxx-sources})
add_dependencies(modulesearchEngine ck)
configure_file(state_space_searchengine/searchEngine.h ${CMAKE_BINARY_DIR}/include COPYONLY)
configure_file(state_space_searchengine/searchEngine_impl.h ${CMAKE_BINARY_DIR}/include COPYONLY)

add_library(modulesearchEngine_bound ${searchengine-h-sources} ${searchengine-cxx-sources} ${CMAKE_BINARY_DIR}/include/searchEngine_bound.decl.h)
add_dependencies(modulesearchEngine_bound ck)
target_compile_options(modulesearchEngine_bound PRIVATE -DBRANCHBOUND)

add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/include/searchEngine_bound.decl.h
      COMMAND sed -e 's/module searchEngine/module searchEngine_bound/' ${CMAKE_CURRENT_SOURCE_DIR}/state_space_searchengine/searchEngine.ci > searchEngine_bound.ci && ${CMAKE_BINARY_DIR}/bin/charmc -DBRANCHBOUND -E searchEngine_bound.ci
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/include/
      DEPENDS charmxi
      )

# pythonCCS

if(CMK_BUILD_PYTHON)
  if(PYTHON_VERSION_MAJOR EQUAL 2) # pythonCCS needs Python 2.x
    set(pythonCCS-h-sources pythonCCS/PythonCCS.h pythonCCS/PythonCCS-client.h)

    file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/include/pythonIncludes.h CONTENT
      "#include \"python${CMK_BUILD_PYTHON}/Python.h\"
       #include \"python${CMK_BUILD_PYTHON}/compile.h\"
       #include \"python${CMK_BUILD_PYTHON}/eval.h\"
       #include \"python${CMK_BUILD_PYTHON}/node.h\"
      "
      )

    foreach(file ${pythonCCS-h-sources})
      configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
    endforeach()

    add_library(modulePythonCCS pythonCCS/PythonCCS.C pythonCCS/PythonCCS-client.C)
    add_library(modulecharmdebug_python pythonCCS/charmdebug-python.C)

    add_dependencies(modulePythonCCS ck)
    add_dependencies(modulecharmdebug_python ck)
  endif()
endif()

# amr
set(amr-h-sources amr/amr.h amr/bitvec.h amr/fifo.h amr/statcoll.h)

set(amr-cxx-sources amr/amr.C amr/fifo.C amr/statcoll.C)

add_library(amr EXCLUDE_FROM_ALL ${amr-cxx-sources} ${amr-h-sources})

add_dependencies(amr ck)

foreach(file ${amr-h-sources})
  configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
endforeach()



# ParFUM

set(parfum-h-sources ParFUM/ParFUM_internals.h ParFUM/ParFUM_locking.h ParFUM/ParFUM_Mesh_Modify.h ParFUM/ParFUM_Adapt_Algs.h ParFUM/ParFUM_Adapt.h ParFUM/ParFUM_Interpolate.h ParFUM/ParFUM_SA.h ParFUM/import.h ParFUM/adapt_adj.h ParFUM/bulk_adapt_ops.h ParFUM/bulk_adapt.h ParFUM/ParFUM.h ParFUM/ParFUMf.h ParFUM/ParFUM_Types.h)

set(parfum-cxx-sources ParFUM/ParFUM.C ParFUM/mesh.C ParFUM/symmetries.C ParFUM/partition.C ParFUM/map.C ParFUM/compat.C ParFUM/call_init.C ParFUM/parallel_part.C ParFUM/mesh_modify.C ParFUM/mesh_adjacency.C ParFUM/adapt.C ParFUM/adapt_algs.C ParFUM/adapt_if.C ParFUM/interpolate.C ParFUM/lock.C ParFUM/util.C ParFUM/lock_node.C ParFUM/adapt_lock.C ParFUM/ParFUM_SA.C ParFUM/global_numbering.C ParFUM/import.C ParFUM/adapt_adj.C ParFUM/bulk_adapt_ops.C ParFUM/bulk_adapt.C ParFUM/mesh_feature_detect.C ParFUM/MsaHashtable.C)

set(parfum-compat-cxx-sources ParFUM/compat_init.C ParFUM/compat_finit.C ParFUM/compat_driver.C ParFUM/compat_fdriver.C)

add_library(moduleParFUM ${parfum-cxx-sources})
add_library(ParFUM-compat ${parfum-compat-cxx-sources})

add_dependencies(moduleParFUM ck)

target_include_directories(moduleParFUM PRIVATE ParFUM/)


# metis

set(metis-cxx-sources metis/libmetis/auxapi.C metis/libmetis/balance.C metis/libmetis/bucketsort.C metis/libmetis/checkgraph.C metis/libmetis/
 metis/libmetis/ metis/libmetis/coarsen.C metis/libmetis/compress.C metis/libmetis/contig.C metis/libmetis/debug.C metis/libmetis/fm.C metis/libmetis/fortran.C metis/libmetis/
 metis/libmetis/ metis/libmetis/frename.C metis/libmetis/gklib.C metis/libmetis/graph.C metis/libmetis/initpart.C metis/libmetis/kmetis.C metis/libmetis/kwayfm.C metis/libmetis/
 metis/libmetis/ metis/libmetis/kwayrefine.C metis/libmetis/mcutil.C metis/libmetis/mesh.C metis/libmetis/meshpart.C metis/libmetis/minconn.C metis/libmetis/mincover.C metis/libmetis/
 metis/libmetis/ metis/libmetis/mmd.C metis/libmetis/ometis.C metis/libmetis/options.C metis/libmetis/parmetis.C metis/libmetis/pmetis.C metis/libmetis/refine.C metis/libmetis/
 metis/libmetis/ metis/libmetis/separator.C metis/libmetis/sfm.C metis/libmetis/srefine.C metis/libmetis/stat.C metis/libmetis/timing.C metis/libmetis/util.C metis/libmetis/wspace.C)


add_library(ckmetis EXCLUDE_FROM_ALL ${metis-cxx-sources})
add_dependencies(ckmetis ck)
configure_file(metis/libmetis/metis.h ${CMAKE_BINARY_DIR}/include COPYONLY)

target_include_directories(ckmetis PRIVATE metis/GKlib .)
target_compile_options(ckmetis PRIVATE -Wno-reserved-user-defined-literal)
