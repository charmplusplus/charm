set(ampi-cxx-sources ampi.C ampiMisc.C ampiOneSided.C ampif.C ddt.C mpich-alltoall.C ampi_mpix.C ampi_noimpl.C)

set(ampi-f90-sources ampifimpl.f90 ampimod.f90)

set(ampi-h-sources ampi.h ampi-interoperate.h ampif.h ampiimpl.h ddt.h ampi_functions.h ampi_funcptr.h ampi_funcptr_loader.h mpio_functions.h mpio.h mpio_functions.h mpio_globals.h mpiof.h)

add_library(ampi-compat compat_ampi.c
       compat_ampim.C compat_ampifm.C compat_ampicm.C
       compat_ampicpp.C)

add_library(moduleampi ${ampi-cxx-sources} ${ampi-h-sources})
add_dependencies(moduleampi moduletcharm ck ampi_funcptr_shim ampi_funcptr_shim_main ampi_funcptr_shim_fortran ampi_funcptr_shim_fmpimain)


# fsglobals/pipglobals
add_library(ampi_funcptr_shim OBJECT ampi_funcptr_shim.C)
target_compile_options(ampi_funcptr_shim PRIVATE -DAMPI_USE_FUNCPTR -fPIE -shared -fvisibility=hidden)

add_library(ampi_funcptr_shim_main OBJECT ampi_funcptr_shim_main.C)
target_compile_options(ampi_funcptr_shim_main PRIVATE -fPIE)

add_library(ampi_funcptr_shim_fortran OBJECT ampif.C)
target_compile_options(ampi_funcptr_shim_fortran PRIVATE -DAMPI_USE_FUNCPTR -fPIE -shared -fvisibility=hidden)

add_library(ampi_funcptr_shim_fmpimain OBJECT ampi_funcptr_shim_fmpimain.C)
target_compile_options(ampi_funcptr_shim_fmpimain PRIVATE -fPIE)

add_library(ampi_funcptr_loader OBJECT ampi_funcptr_loader.C)

add_library(ampi_funcptr_loader_stub OBJECT ampi_funcptr_loader_stub.C)

if(${CMK_SUPPORTS_FSGLOBALS})
    add_library(ampi_globals_fs OBJECT ampi_globals_fs.C)
    add_dependencies(ampi_globals_fs ck)
    add_dependencies(moduleampi ampi_globals_fs)
    add_custom_command(TARGET moduleampi
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_globals_fs.dir/ampi_globals_fs.C.o ${CMAKE_BINARY_DIR}/lib/ampi_globals_fs.o
        VERBATIM
    )
endif()

if(${CMK_SUPPORTS_PIPGLOBALS})
    add_library(ampi_globals_pip OBJECT ampi_globals_pip.C)
    add_dependencies(ampi_globals_pip ck)
    add_dependencies(moduleampi ampi_globals_pip)
    add_custom_command(TARGET moduleampi
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_globals_pip.dir/ampi_globals_pip.C.o ${CMAKE_BINARY_DIR}/lib/ampi_globals_pip.o
        VERBATIM
    )
endif()

add_dependencies(ampi_funcptr_shim ck)
add_dependencies(ampi_funcptr_shim_main ck)
add_dependencies(ampi_funcptr_shim_fortran ck)
add_dependencies(ampi_funcptr_shim_fmpimain ck)
add_dependencies(ampi_funcptr_loader ck)
add_dependencies(ampi_funcptr_loader_stub ck)

add_dependencies(moduleampi ampi_funcptr_shim ampi_funcptr_shim_main ampi_funcptr_shim_fortran ampi_funcptr_shim_fmpimain ampi_funcptr_loader ampi_funcptr_loader_stub)

add_custom_command(TARGET moduleampi
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_shim.dir/ampi_funcptr_shim.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_shim.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_shim_main.dir/ampi_funcptr_shim_main.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_shim_main.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_shim_fortran.dir/ampif.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_shim_fortran.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_shim_fmpimain.dir/ampi_funcptr_shim_fmpimain.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_shim_fmpimain.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_loader.dir/ampi_funcptr_loader.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_loader.o
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampi_funcptr_loader_stub.dir/ampi_funcptr_loader_stub.C.o ${CMAKE_BINARY_DIR}/lib/ampi_funcptr_loader_stub.o
    VERBATIM
)



if(${CMK_CAN_LINK_FORTRAN})
    add_library(moduleampif ${ampi-f90-sources} ${ampi-cxx-sources} ${ampi-h-sources})
    add_dependencies(moduleampif moduletcharm ck)

    add_library(ampif OBJECT ampif.C)
    add_library(ampifimpl OBJECT ampifimpl.f90)
    add_library(ampimod OBJECT ampimod.f90)
    add_dependencies(ampif ck)
    add_dependencies(ampifimpl ck)
    add_dependencies(ampimod ck)
    add_dependencies(moduleampif ampif ampifimpl ampimod)
    add_custom_command(TARGET moduleampif
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampif.dir/ampif.C.o ${CMAKE_BINARY_DIR}/lib/ampif.o
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampifimpl.dir/ampifimpl.f90.o ${CMAKE_BINARY_DIR}/lib/ampifimpl.o
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ampimod.dir/ampimod.f90.o ${CMAKE_BINARY_DIR}/lib/ampimod.o
        VERBATIM
    )
endif()


configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampicc         @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/               @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampicxx        @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/ampic++        @ONLY)
configure_file(ampif77       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampif90       ${CMAKE_BINARY_DIR}/bin/ampifort       COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/               COPYONLY)
configure_file(ampirun       ${CMAKE_BINARY_DIR}/bin/ampiexec       COPYONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpicc.ampi     @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpiCC.ampi     @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpicxx.ampi    @ONLY)
configure_file(ampiCC        ${CMAKE_BINARY_DIR}/bin/mpic++.ampi    @ONLY)
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
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/ampi.h COPYONLY)
  elseif(${file} STREQUAL "ampif.h")
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/mpif.h COPYONLY)
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/ampif.h COPYONLY)
  else()
    configure_file(${file} ${CMAKE_BINARY_DIR}/include COPYONLY)
  endif()
endforeach(file)

configure_file(libmoduleampi.dep  ${CMAKE_BINARY_DIR}/lib COPYONLY)
configure_file(libmoduleampif.dep ${CMAKE_BINARY_DIR}/lib COPYONLY)


# ROMIO
if(${CMK_AMPI_WITH_ROMIO})
    # Compile ROMIO with the same flags as AMPI
    get_target_property(AMPI_CFLAGS moduleampi COMPILE_OPTIONS)
    string(REPLACE ";" " " AMPI_CFLAGS "${AMPI_CFLAGS}")
    set(ROMIO_FLAGS "${AMPI_CFLAGS} -ampi-romio-configure")
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
        CONFIGURE_COMMAND cd ${romio_dir} &&
            export MPI_IMPL=ampi &&
            export FROM_MPICH=no &&
            export FROM_MPICH2=no &&
            export FROM_LAM=no &&
            export F77=${CMAKE_BINARY_DIR}/bin/ampif90 &&
            export CC=${CMAKE_BINARY_DIR}/bin/ampicc &&
            ./configure
            --disable-aio
            --enable-f77=${romio_f77}
            --enable-static=yes
            --enable-shared=${romio_shared}
            --disable-weak-symbols
            CFLAGS=${ROMIO_FLAGS}
            FFLAGS=${ROMIO_FLAGS}
            > /dev/null
        COMMAND cp -f ${romio_dir}/include/mpio.h ${CMAKE_BINARY_DIR}/include/
        COMMAND cp -f ${romio_dir}/include/mpiof.h ${CMAKE_BINARY_DIR}/include/
        COMMAND cp -f ${romio_dir}/include/mpio_functions.h ${CMAKE_BINARY_DIR}/include/
        BUILD_COMMAND make -C ${romio_dir} AUTOCONF=: AUTOHEADER=: AUTOMAKE=: ACLOCAL=: V=$(VERBOSE)
        INSTALL_COMMAND cp -f ${romio_dir}/.libs/libromio.a ${CMAKE_BINARY_DIR}/lib/libampiromio.a
        COMMAND test ${BUILD_SHARED} -eq 1 && cp -f ${romio_dir}/.libs/libromio${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_BINARY_DIR}/lib_so/libampiromio${CMAKE_SHARED_LIBRARY_SUFFIX} || true
        LIST_SEPARATOR ^^
    )

    # Since we link a program with charmc as part of ROMIO's configure,
    # we need all libraries built at this point.
    add_dependencies(romio moduleampi ampi-compat ck modulecompletion
        conv-partition conv-util hwloc conv-static conv-core ldb-rand
        memory-os-isomalloc memory-default threads-default ckmain moduletcharmmain
        conv-machine tmgr conv-ldb ckqt tcharm-compat moduleNDMeshStreamer)
    if(${CMK_CAN_LINK_FORTRAN})
        add_dependencies(romio moduleampif conv-utilf)
    endif()
endif()
