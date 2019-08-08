# cache
add_library(moduleCkCache cache/CkCache.C cache/CkCache.h)
add_dependencies(moduleCkCache ck)

# sparseContiguousReducer
add_library(moduleCkSparseContiguousReducer sparseContiguousReducer/cksparsecontiguousreducer.h sparseContiguousReducer/cksparsecontiguousreducer.C)
add_dependencies(moduleCkSparseContiguousReducer ck)

# multiphaseSharedArrays
add_library(modulemsa multiphaseSharedArrays/msa.h multiphaseSharedArrays/msa-common.h multiphaseSharedArrays/msa-distArray.h multiphaseSharedArrays/msa-DistPageMgr.h multiphaseSharedArrays/msa-distArray.C)
add_dependencies(modulemsa ck)

# armci
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

add_library(libmodulemblock mblock/mblock.C mblock/patch.C
mblock/compat_driver.c mblock/compat_fdriver.c mblock/mblock.h
mblock/mblockf.h mblock/patch.h mblock/gridutil.h mblock/mblock_impl.h)
add_dependencies(libmodulemblock ck)

# barrier
configure_file(barrier/barrier.h  ${CMAKE_BINARY_DIR}/include COPYONLY)
add_library(libmodulebarrier barrier/barrier.C barrier/barrier.h)
add_dependencies(libmodulebarrier ck)

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

target_compile_options(moduleliveViz PRIVATE -DEXTERIOR_BLACK_PIXEL_ELIMINATION)

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
