set(ampi-cxx-sources ampi.C ampiMisc.C ampiOneSided.C ampif.C ddt.C mpich-alltoall.C ampi_mpix.C)

set(ampi-f90-sources ampifimpl.f90 ampimod.f90)

set(ampi-h-sources ampi.h ampif.h ampiimpl.h ddt.h)

include(../../../ci-files.cmake)

add_library(ampi-compat compat_ampius.C compat_ampifus.C compat_ampi.c
       compat_ampim.C compat_ampifm.C compat_ampicm.C
       compat_ampicpp.C)

add_library(moduleampi ${ampi-cxx-sources} ${ampi-h-sources} ${CMAKE_BINARY_DIR}/include/ampi.decl.h)
add_dependencies(moduleampi moduletcharm)

if(${CMK_CAN_LINK_FORTRAN})
    add_library(moduleampif ${ampi-f90-sources})
    add_dependencies(moduleampi moduleampif)
endif()


configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/           COPYONLY)
configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/ampicc     COPYONLY)
configure_file(ampiCC  ${CMAKE_BINARY_DIR}/bin/ampicxx    COPYONLY)
configure_file(ampif77 ${CMAKE_BINARY_DIR}/bin            COPYONLY)
configure_file(ampif90 ${CMAKE_BINARY_DIR}/bin            COPYONLY)
configure_file(ampirun ${CMAKE_BINARY_DIR}/bin            COPYONLY)

# copy these files after the build in order not to interfere when building
# charm++ on MPI
add_custom_command(TARGET moduleampi POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ampi.h ${CMAKE_BINARY_DIR}/include/mpi.h
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ampif.h ${CMAKE_BINARY_DIR}/include/mpif.h
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/mpio.h ${CMAKE_BINARY_DIR}/include/
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/mpiof.h ${CMAKE_BINARY_DIR}/include/
)

configure_file(libmoduleampi.dep  ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(libmoduleampif.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)


# ROMIO
if(${CMK_AMPI_WITH_ROMIO})
    include(ExternalProject)
    ExternalProject_Add(romio
        URL ${CMAKE_CURRENT_SOURCE_DIR}/romio
        CONFIGURE_COMMAND cd ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/ && ./configure
            -mpiincdir=${CMAKE_BINARY_DIR}/include
            -mpilib=${CMAKE_BINARY_DIR}/lib/libmoduleampi.a
            -mpibindir=${CMAKE_BINARY_DIR}/bin
            -cc=${CMAKE_BINARY_DIR}/bin/ampicc
            -fc=${CMAKE_BINARY_DIR}/bin/ampif77
            -f90=${CMAKE_BINARY_DIR}/bin/ampif90
            # -cflags="$(OPTS)"
            # -fflags="$(OPTS)"
            -noprofile
            -mpiolib=${CMAKE_BINARY_DIR}/lib/libampiromio.a > /dev/null
        COMMAND cp -f  ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/include/mpio.h ${CMAKE_BINARY_DIR}/include/
        COMMAND cp -f  ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/include/mpiof.h ${CMAKE_BINARY_DIR}/include/
        BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio/ && make
        COMMAND # build shared library
            find .. -name *.o -exec ${CMAKE_CXX_COMPILER_LAUNCHER} -shared -o ${CMAKE_BINARY_DIR}/lib_so/libampiromio${CMAKE_SHARED_LIBRARY_SUFFIX} {} + ;
        INSTALL_COMMAND "" # No install step
        LIST_SEPARATOR ^^
    )

    # Since we link a program with charmc as part of ROMIO's configure,
    # we need all libraries built at this point.
    add_dependencies(romio moduleampi ampi-compat moduleampif ck modulecompletion conv-partition conv-util)
endif()
