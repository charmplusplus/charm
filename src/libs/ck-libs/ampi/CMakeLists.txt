set(ampi-cxx-sources ampi.C ampiMisc.C ampiOneSided.C ampif.C ddt.C mpich-alltoall.C ampi_mpix.C)

set(ampi-f90-sources ampifimpl.f90 ampimod.f90)

set(ampi-h-sources ampi.h ampif.h ampiimpl.h ddt.h ampi_functions.h ampi_funcptr.h ampi_funcptr_loader.h mpio_functions.h mpio.h mpio_functions.h mpio_globals.h mpiof.h)

include(${CMAKE_SOURCE_DIR}/cmake/ci-files.cmake)

add_library(ampi-compat compat_ampi.c
       compat_ampim.C compat_ampifm.C compat_ampicm.C
       compat_ampicpp.C)

add_library(moduleampi ${ampi-cxx-sources} ${ampi-h-sources})
add_dependencies(moduleampi moduletcharm ck)


# fsglobals/pipglobals
add_library(ampi_funcptr_shim OBJECT ampi_funcptr_shim.C)
target_compile_options(ampi_funcptr_shim PRIVATE -ampi-funcptr-shim)

add_library(ampi_funcptr_fortran OBJECT ampif.C)
target_compile_options(ampi_funcptr_fortran PRIVATE -ampi-funcptr-shim)

add_library(ampi_funcptr_loader OBJECT ampi_funcptr_loader.C)
add_library(ampi_funcptr_fsglobals OBJECT ampi_funcptr_fsglobals.C)
add_library(ampi_funcptr_pipglobals OBJECT ampi_funcptr_pipglobals.C)

add_custom_command(TARGET moduleampi
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_shim.dir/ampi_funcptr_shim.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_shim.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_fortran.dir/ampif.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_fortran.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_loader.dir/ampi_funcptr_loader.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_loader.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_fsglobals.dir/ampi_funcptr_fsglobals.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_fsglobals.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_pipglobals.dir/ampi_funcptr_pipglobals.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_pipglobals.o
    VERBATIM
)



if(${CMK_CAN_LINK_FORTRAN})
    add_library(moduleampif ${ampi-f90-sources} ${ampi-cxx-sources} ${ampi-h-sources})
    add_dependencies(moduleampif moduletcharm ck)
endif()


configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampicc         COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampicxx        COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampic++        COPYONLY)
configure_file(ampif77       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/ampifort       COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/ampiexec       COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpicc.ampi     COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpiCC.ampi     COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpicxx.ampi    COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpic++.ampi    COPYONLY)
configure_file(ampif77       ${CMAKE_BINARY_DIR}/bin/mpif77.ampi    COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/mpif90.ampi    COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/mpifort.ampi   COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/mpirun.ampi    COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/mpiexec.ampi   COPYONLY)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ampi/)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpicc     COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpiCC     COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpicxx    COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpic++    COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpif77    COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpif90    COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpifort   COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpirun    COPYONLY)
configure_file(pathstub.sh   ${CMAKE_BINARY_DIR}/bin/ampi/mpiexec   COPYONLY)

configure_file(nm_globals.sh ${CMAKE_BINARY_DIR}/bin/               COPYONLY)

foreach(file ${ampi-h-sources})
  if(${file} STREQUAL "ampi.h")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/ampi/)
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
    if(${CMK_CAN_LINK_FORTRAN})
        set(romio_f77 yes)
    else()
        set(romio_f77 no)
    endif()
    if(${BUILD_SHARED})
        set(romio_shared yes)
    else()
        set(romio_shared no)
    endif()
    set(romio_dir ${CMAKE_BINARY_DIR}/src/libs/ck-libs/ampi/romio-prefix/src/romio)
    include(ExternalProject)
    ExternalProject_Add(romio
        URL ${CMAKE_CURRENT_SOURCE_DIR}/romio
        # touch: update timestamps on these files so that ROMIO's Makefile does not run configure a second time
        CONFIGURE_COMMAND cd ${romio_dir} &&
            touch aclocal.m4 Makefile.in configure config.status &&
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
            --enable-f77=${romio_f77}
            --enable-static=yes
            --enable-shared=${romio_shared}
            --disable-weak-symbols
             > /dev/null
        COMMAND cp -f ${romio_dir}/include/mpio.h ${CMAKE_BINARY_DIR}/include/
        COMMAND cp -f ${romio_dir}/include/mpiof.h ${CMAKE_BINARY_DIR}/include/
        BUILD_COMMAND cd ${romio_dir} && make > /dev/null
        COMMAND cp -f ${romio_dir}/.libs/libromio.a ${CMAKE_BINARY_DIR}/lib/libampiromio.a
        COMMAND test ${BUILD_SHARED} -eq 1 && cp -f ${romio_dir}/.libs/libromio${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_BINARY_DIR}/lib_so/libampiromio${CMAKE_SHARED_LIBRARY_SUFFIX} || true
        INSTALL_COMMAND "" # No install step
        LIST_SEPARATOR ^^
    )

    # Since we link a program with charmc as part of ROMIO's configure,
    # we need all libraries built at this point.
    add_dependencies(romio moduleampi ampi-compat ck modulecompletion conv-partition conv-util hwloc conv-static conv-core)
    if(${CMK_CAN_LINK_FORTRAN})
        add_dependencies(romio moduleampif)
    endif()
endif()
