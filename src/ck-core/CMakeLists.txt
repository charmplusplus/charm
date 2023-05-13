set(ck-cxx-sources
    ../ck-perf/trace-common.C ../ck-perf/trace-controlPoints.C
    ../ck-perf/tracec.C ../ck-perf/tracef.C ../conv-perf/charmProjections.C
    ../util/ckbitvector.C ../util/ckregex.C
    ck.C ckarray.C ckarrayoptions.C ckcallback.C
    ckcheckpoint.C ckfutures.C ckIgetControl.C
    cklocation.C ckmemcheckpoint.C ckmulticast.C
    ckobjQ.C ckrdma.C ckrdmadevice.C ckreduction.C cksyncbarrier.C debug-charm.C
    debug-message.C init.C modifyScheduler.C mpi-interoperate.C msgalloc.C
    qd.C register.C sdag.C waitqd.C
)

set(ck-h-sources XArraySectionReducer.h charm++.h charm++_type_traits.h
    charm-api.h charm.h charmf.h ck.h ckIgetControl.h ckarray.h ckarrayindex.h
    ckarrayoptions.h ckcallback-ccs.h ckcallback.h ckcheckpoint.h
    ckmarshall.h ckfutures.h cklocation.h cklocrec.h
    ckmemcheckpoint.h ckmessage.h ckmigratable.h ckmulticast.h
    ckobjQ.h ckrdma.h ckrdmadevice.h ckreduction.h cksection.h
    ckstream.h cksyncbarrier.h debug-charm.h envelope-path.h envelope.h init.h
    middle-conv.h middle.h mpi-interoperate.h objid.h qd.h
    readonly.h register.h sdag.h stats.h waitqd.h)

set(ldb-cxx-sources)

set(ldb-h-sources
    ../util/json.hpp ../util/json_fwd.hpp)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

add_library(ck ${ck-cxx-sources} ${ck-h-sources} ${all-ci-outputs})
add_library(ckmain main.C)
add_dependencies(ckmain ck)

# Single LBs

foreach(filename ${ck-h-sources} ${ldb-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

target_include_directories(ck PRIVATE . ../ck-perf ../util/topomanager ../conv-perf)
target_include_directories(ckmain PRIVATE . ../ck-perf ../util/topomanager)

# Fortran interface
if(CMK_CAN_LINK_FORTRAN)
    add_library(ckf charmf.C charmmod.f90 charmf.h)
    add_dependencies(ckf ck)
    add_custom_command(TARGET ckf
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/charm.mod ${CMAKE_BINARY_DIR}/include/
        VERBATIM
    )
    add_library(ckmainf mainf.f90)

    add_library(f90charm ../langs/f90charm/f90main.C ${CMAKE_BINARY_DIR}/include/f90main.decl.h)
    add_dependencies(f90charm ck)
endif()

add_library(mpi-mainmodule mpi-mainmodule.C)
add_dependencies(mpi-mainmodule ck)
