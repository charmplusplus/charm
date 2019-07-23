set(ampi-cxx-sources ampi.C ampiMisc.C ampiOneSided.C ampif.C ddt.C mpich-alltoall.C ampi_mpix.C)

set(ampi-f90-sources ampifimpl.f90 ampimod.f90)

set(ampi-h-sources ampi.h ampif.h ampiimpl.h ddt.h ampi_functions.h ampi_funcptr.h ampi_funcptr_loader.h mpio_functions.h mpio.h mpio_functions.h mpiof.h)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

add_library(ampi-compat compat_ampius.C compat_ampifus.C compat_ampi.c
       compat_ampim.C compat_ampifm.C compat_ampicm.C
       compat_ampicpp.C)

add_library(moduleampi ${ampi-cxx-sources} ${ampi-h-sources})
add_dependencies(moduleampi moduletcharm ck)

if(${CMK_CAN_LINK_FORTRAN})
    add_library(moduleampif ${ampi-f90-sources} ${ampi-cxx-sources} ${ampi-h-sources})
    add_dependencies(moduleampif moduletcharm ck)
endif()

# FIXME: add fsglobls/pipglobals

configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/           COPYONLY)
configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/ampicc     COPYONLY)
configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/ampicxx    COPYONLY)
configure_file(ampif77 ${CMAKE_BINARY_DIR}/bin            COPYONLY)
configure_file(ampif90 ${CMAKE_BINARY_DIR}/bin            COPYONLY)
configure_file(ampirun ${CMAKE_BINARY_DIR}/bin            COPYONLY)

foreach(file ${ampi-h-sources})
  if(${file} STREQUAL "ampi.h")
    make_directory(${CMAKE_BINARY_DIR}/include/ampi/)
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/ampi/mpi.h COPYONLY)
  elseif(${file} STREQUAL "ampif.h")
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/mpif.h COPYONLY)
  else()
    configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
  endif()
endforeach(file)

configure_file(libmoduleampi.dep  ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(libmoduleampif.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)


# ROMIO
if(${CMK_AMPI_WITH_ROMIO})
    include(ExternalProject)
    ExternalProject_Add(romio
        URL ${CMAKE_CURRENT_SOURCE_DIR}/romio
        CONFIGURE_COMMAND cd ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/ &&
            export MPI_IMPL=ampi &&
            export FROM_MPICH=no &&
            export FROM_MPICH2=no && 
            export FROM_LAM=no &&
            export CFLAGS=-ampi-romio-configure &&
            export FFLAGS=-ampi-romio-configure &&
            export F77=${CMAKE_BINARY_DIR}/bin/ampif90 &&
            export CC=${CMAKE_BINARY_DIR}/bin/ampicc &&
            ./configure
            --disable-aio
            --enable-f77=${CMK_CAN_LINK_FORTRAN}
             > /dev/null
        COMMAND cp -f  ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/include/mpio.h ${CMAKE_BINARY_DIR}/include/
        COMMAND cp -f  ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/include/mpiof.h ${CMAKE_BINARY_DIR}/include/
        BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/ && make > /dev/null
        COMMAND # build shared library
            test ${BUILD_SHARED} -eq 1 && find .. -name *.o -exec ${CMAKE_CXX_COMPILER_LAUNCHER} -shared -o ${CMAKE_BINARY_DIR}/lib_so/libampiromio${CMAKE_SHARED_LIBRARY_SUFFIX} {} + ; || true
        INSTALL_COMMAND "" # No install step
        LIST_SEPARATOR ^^
    )

    # Since we link a program with charmc as part of ROMIO's configure,
    # we need all libraries built at this point.
    add_dependencies(romio moduleampi ampi-compat moduleampif ck modulecompletion conv-partition conv-util hwloc)
endif()
