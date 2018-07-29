cmake_minimum_required(VERSION 3.8)

project(Charm++ LANGUAGES CXX C VERSION 6.9.0)

find_package(Threads REQUIRED)
find_package(OpenMP) # Do this before Fortran, in case we don't have a Fortran compiler
find_package(ZLIB)


file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib_so/)
file(TOUCH          ${CMAKE_BINARY_DIR}/lib_so/.charmso)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/)

# We need C++11 for (almost) all targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(Fortran OPTIONAL)
include(FortranCInterface)
FortranCInterface_VERIFY()
FortranCInterface_VERIFY(CXX)

if(${FortranCInterface_VERIFIED_CXX})
  set(CMK_CAN_LINK_FORTRAN 1)
else()
  set(CMK_CAN_LINK_FORTRAN 0)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(CMK_COMPILER gcc)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMK_COMPILER clang)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set(CMK_COMPILER icc)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Cray")
  set(CMK_COMPILER craycc)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
  set(CMK_COMPILER pgcc)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "XL")
  set(CMK_COMPILER xlc)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(CMK_COMPILER msvc)
else()
  message(FATAL_ERROR "Unknown compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

# Build options
set(NETWORK "netlrts" CACHE STRING "Target network layer, can be one of mpi,multicore,netlrts,gni,ofi,pami,pamilrts,verbs")

# Platform specific options (choose multiple if desired):
option(SMP "Enable shared-memory communication within a single multi-processor machine instead of message passing." ON)
option(OMP "Enable OpenMP support in Charm++" OFF)
option(TCP "use TCP sockets for communication (only for netlrts)" OFF)
option(PTHREADS "compile with pthreads Converse threads" OFF)

# Advanced options:
option(BIGEMULATOR "Compile for BigSim simulator" OFF)
option(OOC "Compile with out of core support" OFF)
option(SYNCFT "Compile with Charm++ fault tolerance support" OFF)
option(MLOGFT "Compile with Charm++ message logging fault tolerance support" OFF)
option(PAPI "Compile with PAPI performance counter support" OFF)

# Enable/disable features:
option(ERROR_CHECKING "Enable error checking" OFF)
option(AMPI_ERROR_CHECKING "Enable AMPI error checking" OFF)
option(STATS               "Enable statistics collection" OFF)
option(TRACING             "Enable tracing modules" OFF)
option(TRACING_COMMTHREAD  "Enable tracing communication thread" OFF)
option(CHARMDEBUG          "Enable charmDebug" OFF)
option(REPLAY              "Enable record/replay" OFF)
option(CCS                 "Enable CCS" OFF)
option(CONTROLPOINT        "Enable control point" OFF)
option(LBUSERDATA          "Enable LB user data" OFF)
option(LOCKLESS_QUEUE      "Enable lockless queue for PE local and node queue" OFF)
option(SHRINKEXPAND        "Enable malleable jobs / shrink expand" OFF)
option(CHARMPY             "Enable CharmPy support" OFF)
option(NUMA                "support memory affinity with NUMA" OFF)
set(LBTIME_TYPE "double" CACHE STRING "Load balancing timer type")
option(QLOGIC              "QLogic based Infiniband" OFF)
set(REFNUM_TYPE "unsigned short" CACHE STRING  "Size of the envelope refnum field")
set(PRIO_TYPE "bitvec"    CACHE STRING "Size of expected message priorities")
option(RANDOMIZED_MSGQ     "Enable a randomized msg queue (for debugging etc)" OFF)
option(ZLIB                "Enable zlib support" OFF)

# Other options
option(EVERYLB             "Compile EveryLB suite of load balancing strategies" OFF)
option(F90CHARM            "Build f90charm" OFF)

if(${OMP})
    find_package(OpenMP REQUIRED)
endif()

if(${NETWORK} STREQUAL "multicore")
  set(SMP 0)
endif()

set(CMK_USE_ZLIB ${ZLIB})
set(CMK_REFNUM_TYPE ${REFNUM_TYPE})
set(CMK_LBTIME_TYPE ${LBTIME_TYPE})
set(CMK_LB_USER_DATA 0)

set(CMK_RANDOMIZED_MSGQ ${RANDOMIZED_MSGQ})
set(CMK_SHRINK_EXPAND ${SHRINKEXPAND})
set(CMK_SMP_TRACE_COMMTHREAD ${TRACING_COMMTHREAD})

set(CMK_TRACE_ENABLED ${TRACING})

set(CMK_WITH_CONTROLPOINT ${CONTROLPOINT})
set(CMK_WITH_STATS ${STATS})

set(CMK_CHARMDEBUG ${CHARMDEBUG})
set(CMK_CHARMPY ${CHARMPY})


set(CMK_MSG_PRIO_TYPE ${PRIO_TYPE})
if(${CMK_MSG_PRIO_TYPE} STREQUAL "bitvec")
  set(CMK_USE_STL_MSGQ 0)
else()
  set(CMK_USE_STL_MSGQ 1)
endif()

set(CMK_OS_IS_LINUX 0)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CHARM_OS "darwin")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CHARM_OS "win")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CHARM_OS "linux")
  set(CMK_OS_IS_LINUX 1)
else()
  message(FATAL_ERROR "Unsupported operating system ${CMAKE_SYSTEM_NAME}.")
endif()

set(CMK_POWER7 0)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CHARM_CPU "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "powerpc" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc")
  set(CHARM_CPU "ppc")
  execute_process(COMMAND grep f Makefile RESULT_VARIABLE isppc7 OUTPUT_QUIET ERROR_QUIET)
  if(${isppc7})
    set(CMK_POWER7 1)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm7")
  set(CHARM_CPU "arm7")
else()
  message(FATAL_ERROR "Unsupported CPU ${CMAKE_SYSTEM_PROCESSOR}.")
endif()


set(CHARM_PLATFORM "${NETWORK}-${CHARM_OS}-${CHARM_CPU}")

set(CMK_BUILD_MPI 0)
set(CMK_BUILD_ON_MPI 0)
if(${NETWORK} STREQUAL "mpi")
  set(CMK_BUILD_MPI 1)
  set(CMK_BUILD_ON_MPI 1)
  find_package(MPI REQUIRED)
endif()

set(CMK_BUILD_OFI 0)
if(${NETWORK} STREQUAL "ofi")
  set(CMK_BUILD_OFI 1)
endif()

set(CMK_BUILD_CRAY 0)
set(CMK_ERROR_CHECKING ${ERROR_CHECKING})
set(CMK_LOCKLESS_QUEUE 0)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib_so)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ./install)
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "${PROJECT_NAME}: No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMK_OPTIMIZE 0)
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(CMK_OPTIMIZE 1)
    add_compile_options(-optimize -production)
endif()

include(src/arch/detect-features.cmake)

configure_file(${CMAKE_SOURCE_DIR}/src/scripts/charmc ${CMAKE_BINARY_DIR}/bin COPYONLY)

# Use charmc to compile/link everything
set(CMAKE_C_COMPILER_LAUNCHER ${CMAKE_BINARY_DIR}/bin/charmc)
set(CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_BINARY_DIR}/bin/charmc)
set(CMAKE_Fortran_COMPILER_LAUNCHER ${CMAKE_BINARY_DIR}/bin/charmc)
set(CMAKE_CXX_LINK_EXECUTABLE
      "${CMAKE_BINARY_DIR}/bin/charmc <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")

set(CMAKE_CXX_CREATE_SHARED_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc -build-shared <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")

set(CMAKE_C_CREATE_SHARED_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc -build-shared <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_C_CREATE_STATIC_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")

set(CMAKE_Fortran_CREATE_SHARED_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc -build-shared <CMAKE_Fortran_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_Fortran_CREATE_STATIC_LIBRARY
      "${CMAKE_BINARY_DIR}/bin/charmc <CMAKE_Fortran_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")


set(CHARM_VERSION ${PROJECT_VERSION})
set(CMK_COMPILER_KNOWS_C11 1)
set(CMK_COMPILER_KNOWS_CPP11 1)
set(CMK_AMPI_WITH_ROMIO 1)

set(CMK_ENABLE_C11   ${CMAKE_C11_EXTENSION_COMPILE_OPTION})
set(CMK_ENABLE_CPP11 ${CMAKE_CXX11_STANDARD_COMPILE_OPTION})
set(OPTS_CXX ${CMAKE_CXX11_STANDARD_COMPILE_OPTION})

set(CMK_CC ${CMAKE_C_COMPILER})
set(CMK_CXX ${CMAKE_CXX_COMPILER})
set(CMK_CF90 ${CMAKE_Fortran_COMPILER})

set(CMK_C_OPENMP ${OpenMP_C_FLAGS})
set(CMK_F_OPENMP ${OpenMP_Fortran_FLAGS})

# Shell scripts

if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${NETWORK}/gdir_link)
  file(STRINGS src/arch/${NETWORK}/gdir_link GDIR)
else()
  set(GDIR ${NETWORK})
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${CHARM_PLATFORM}/vdir_link)
  file(STRINGS src/arch/${CHARM_PLATFORM}/vdir_link VDIR)
else()
  set(VDIR ${CHARM_PLATFORM})
endif()

# configure_file(src/arch/common/conv-mach-opt.sh            include/ COPYONLY) # generated
# configure_file(src/arch/common/conv-mach-pre.sh            include/ COPYONLY) # unused
configure_file(src/arch/common/cc-clang.sh                   include/ COPYONLY)
configure_file(src/arch/common/cc-clang.h                    include/ COPYONLY)
configure_file(src/arch/common/cc-gcc.sh                     include/ COPYONLY)
configure_file(src/arch/common/cc-gcc.h                      include/ COPYONLY)
configure_file(src/arch/common/cc-mpi.sh                     include/ COPYONLY)
configure_file(src/arch/common/cc-mpiopts.sh                 include/ COPYONLY) #sh only
configure_file(src/arch/common/conv-mach-bigemulator.sh      include/ COPYONLY)
configure_file(src/arch/common/conv-mach-bigsim.sh           include/ COPYONLY)
configure_file(src/arch/common/conv-mach-causalft.sh         include/ COPYONLY)
configure_file(src/arch/common/conv-mach-craype.sh           include/ COPYONLY)
configure_file(src/arch/common/conv-mach-cuda.sh             include/ COPYONLY)
configure_file(src/arch/common/conv-mach-darwin.sh           include/ COPYONLY)
configure_file(src/arch/common/conv-mach-flang.sh            include/ COPYONLY)
configure_file(src/arch/common/conv-mach-gfortran.sh         include/ COPYONLY)
configure_file(src/arch/common/conv-mach-ifort.sh            include/ COPYONLY)
configure_file(src/arch/common/conv-mach-mlogft.sh           include/ COPYONLY)
configure_file(src/arch/common/conv-mach-nolb.sh             include/ COPYONLY)
configure_file(src/arch/common/conv-mach-omp.sh              include/ COPYONLY)
configure_file(src/arch/common/conv-mach-ooc.sh              include/ COPYONLY)
configure_file(src/arch/common/conv-mach-papi.sh             include/ COPYONLY)
configure_file(src/arch/common/conv-mach-pedantic.sh         include/ COPYONLY)
configure_file(src/arch/common/conv-mach-perftools.sh        include/ COPYONLY)
configure_file(src/arch/common/conv-mach-persistent.sh       include/ COPYONLY)
configure_file(src/arch/common/conv-mach-pgf90.sh            include/ COPYONLY)
configure_file(src/arch/common/conv-mach-syncft.sh           include/ COPYONLY)
configure_file(src/arch/common/conv-mach-tsan.sh             include/ COPYONLY)
configure_file(src/scripts/conv-config.sh                    include/ COPYONLY)
configure_file(src/arch/${VDIR}/conv-mach.sh                 include/ COPYONLY)

if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${VDIR}/conv-mach-pxshm.sh)
  configure_file(src/arch/${VDIR}/conv-mach-pxshm.sh         include/ COPYONLY)
  configure_file(src/arch/${VDIR}/conv-mach-pxshm.h          include/ COPYONLY)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${VDIR}/conv-mach-smp.sh)
  configure_file(src/arch/${VDIR}/conv-mach-smp.sh           include/ COPYONLY)
  configure_file(src/arch/${VDIR}/conv-mach-smp.h            include/ COPYONLY)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${GDIR}/conv-mach-tcp.sh)
  configure_file(src/arch/${GDIR}/conv-mach-tcp.sh           include/ COPYONLY)
  configure_file(src/arch/${GDIR}/conv-mach-tcp.h            include/ COPYONLY)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${GDIR}/conv-mach-local.sh)
  configure_file(src/arch/${GDIR}/conv-mach-local.sh           include/ COPYONLY)
  configure_file(src/arch/${GDIR}/conv-mach-local.h            include/ COPYONLY)
endif()

# Header files
configure_file(src/arch/${GDIR}/conv-common.h                include/ COPYONLY)
configure_file(src/arch/${VDIR}/conv-mach.h                  include/ COPYONLY)
configure_file(src/arch/util/lrts-common.h                   include/ COPYONLY)
configure_file(src/arch/util/cmiqueue.h                      include/ COPYONLY)
configure_file(src/arch/util/lrtslock.h                      include/ COPYONLY)
configure_file(src/arch/util/mempool.h                       include/ COPYONLY)
configure_file(src/arch/util/machine.h                       include/ COPYONLY)
configure_file(src/arch/util/machine-lrts.h                  include/ COPYONLY)
configure_file(src/arch/util/machine-smp.h                   include/ COPYONLY)
configure_file(src/arch/util/pcqueue.h                       include/ COPYONLY)

set(src-util-h-sources src/util/SSE-Double.h src/util/SSE-Float.h
    src/util/ck128bitHash.h src/util/ckBIconfig.h src/util/ckbitvector.h
    src/util/ckcomplex.h src/util/ckdll.h src/util/ckhashtable.h
    src/util/ckimage.h src/util/cklists.h src/util/ckliststring.h
    src/util/ckregex.h src/util/cksequence.h src/util/cksequence_factory.h
    src/util/cksequence_internal.h src/util/ckstatistics.h src/util/ckvector3d.h
    src/util/cmimemcpy.h src/util/cmimemcpy_qpx.h src/util/cmirdmautils.h
    src/util/cmitls.h src/util/conv-lists.h src/util/crc32.h src/util/hilbert.h
    src/util/partitioning_strategies.h src/util/pup.h src/util/pup_c.h
    src/util/pup_cmialloc.h src/util/pup_mpi.h src/util/pup_paged.h
    src/util/pup_stl.h src/util/pup_toNetwork.h src/util/pup_toNetwork4.h
    src/util/pupf.h src/util/rand48_replacement.h src/util/random_sequence.h
    src/util/simd.h src/util/sockRoutines.h src/util/spanningTree.h
    src/util/spanningTreeStrategy.h src/util/spanningTreeVertex.h
    src/util/strided_sequence.h src/util/treeStrategy_3dTorus_minBytesHops.h
    src/util/treeStrategy_3dTorus_minHops.h
    src/util/treeStrategy_nodeAware_minBytes.h
    src/util/treeStrategy_nodeAware_minGens.h src/util/treeStrategy_topoUnaware.h
    src/util/uFcontext.h src/util/uJcontext.h src/util/valgrind.h
    src/util/vector2d.h)

foreach(filename ${src-util-h-sources})
    configure_file(${filename} include/ COPYONLY)
endforeach()


# Charmxi
add_subdirectory(src/xlat-i)

# Converse
add_subdirectory(src/QuickThreads)
add_subdirectory(src/util/boost-context)
add_subdirectory(src/conv-core)
add_subdirectory(src/conv-ldb)
add_subdirectory(src/util)


# Hwloc
# FIXME: copy .so
include(ExternalProject)
ExternalProject_Add(hwloc
    URL ${CMAKE_SOURCE_DIR}/contrib/hwloc
    CONFIGURE_COMMAND cd ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc && autoreconf -ivf 2> /dev/null 1> /dev/null
    COMMAND ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc/configure
        --disable-cairo
        --disable-cuda
        --disable-gl
        --disable-libnuma
        --disable-libxml2
        --disable-nvml
        --disable-opencl
        --disable-pci
        --disable-libudev
        --disable-visibility
        --enable-embedded-mode
        --enable-shared
        --enable-static
        --with-hwloc-symbol-prefix=cmi_
        --without-x
        > /dev/null
    BUILD_COMMAND make > /dev/null
    COMMAND cp src/.libs/libhwloc_embedded.a ${CMAKE_BINARY_DIR}/lib/
    COMMAND cp -f ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc/include/hwloc.h ${CMAKE_BINARY_DIR}/include/
    COMMAND cp -LRf ${CMAKE_SOURCE_DIR}/contrib/hwloc/include/hwloc ${CMAKE_BINARY_DIR}/include/
    COMMAND cp -f ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/include/hwloc/autogen/config.h ${CMAKE_BINARY_DIR}/include/hwloc/autogen/
)

# TopoManager
set(tmgr-cxx-sources src/util/topomanager/TopoManager.C)
set(tmgr-h-sources src/util/topomanager/TopoManager.h ${CMAKE_BINARY_DIR}/include/topomanager_config.h)
file(WRITE ${CMAKE_BINARY_DIR}/include/topomanager_config.h "// empty\n" )
add_library(tmgr ${tmgr-cxx-sources} ${tmgr-h-sources})
foreach(filename ${tmgr-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

# Charm++
add_subdirectory(src/ck-core)
add_subdirectory(src/ck-perf)
add_subdirectory(src/ck-pics)


set(modulecompletion-h-sources src/libs/ck-libs/completion/completion.h)
set(modulecompletion-cxx-sources src/libs/ck-libs/completion/completion.C)
add_library(modulecompletion ${modulecompletion-cxx-sources} ${modulecompletion-h-sources} )
configure_file(src/libs/ck-libs/completion/completion.h include/ COPYONLY)
add_dependencies(modulecompletion ck)

# Charmrun + testrun
if(EXISTS ${CMAKE_SOURCE_DIR}/src/arch/${NETWORK}/charmrun)
  configure_file(src/arch/${NETWORK}/charmrun ${CMAKE_BINARY_DIR}/bin COPYONLY)
else()
  add_subdirectory(src/util/charmrun-src)
  add_dependencies(charmrun create_symlinks)
endif()
configure_file(src/scripts/testrun bin/ COPYONLY)

# Charm++ libraries
add_subdirectory(src/libs/ck-libs/NDMeshStreamer)
add_subdirectory(src/libs/ck-libs/tcharm)
add_subdirectory(src/libs/ck-libs/ampi)
add_subdirectory(src/libs/ck-libs/ckloop)

# Create conv-mach-opt.sh
set(optfile_sh ${CMAKE_BINARY_DIR}/include/conv-mach-opt.sh)

file(WRITE  ${optfile_sh} "# Automatically build-time generated options header\n")
if(NOT ${NETWORK} STREQUAL "mpi")
    file(APPEND ${optfile_sh} ". ${CMAKE_BINARY_DIR}/include/cc-${CMK_COMPILER}.sh\n")
endif()
file(APPEND ${optfile_sh} "CMK_NO_BUILD_SHARED=false\n")

foreach(l CMK_COMPILER_KNOWS_C11 CMK_COMPILER_KNOWS_CPP11 CMK_ENABLE_C11 CMK_ENABLE_CPP11 CMK_C_OPENMP OPTS_CXX CMK_AMPI_WITH_ROMIO)
    file(APPEND ${optfile_sh} "${l}=\"${${l}}\"\n" )
endforeach(l)

foreach(l CMK_CC CMK_CXX CMK_CF90)
    file(APPEND ${optfile_sh} "[ -z \"$${l}\" ] && ${l}=\"${${l}}\"\n" )
endforeach()

file(APPEND ${optfile_sh} "CMK_HWLOC_LIBS=\"-lhwloc_embedded\"\n" )
file(APPEND ${optfile_sh} "CMK_LIBS=\"$CMK_LIBS -lz\"\n")


# Create conv-mach-opt.h
set(optfile_h ${CMAKE_BINARY_DIR}/include/conv-mach-opt.h)

file(WRITE  ${optfile_h} "/* Automatically build-time generated options header */\n")
file(APPEND ${optfile_h} "#define CMK_OPTIMIZE ${CMK_OPTIMIZE}\n")

if(NOT ${NETWORK} STREQUAL "mpi")
    file(APPEND ${optfile_h} "#include \"cc-${CMK_COMPILER}.h\"\n")
endif()

# Add options
foreach(opt SMP OMP TCP PTHREADS)
    if(${opt})
        string(TOLOWER ${opt} optl)
        file(APPEND ${optfile_sh} ". ${CMAKE_BINARY_DIR}/include/conv-mach-${optl}.sh\n")
        file(APPEND ${optfile_h} "#include \"conv-mach-${optl}.h\"\n")
        set(opts_enabled "${opts_enabled} ${opt}")
    endif()
endforeach()

# Print configuration
message("==============================")
message("Charm++ ${CHARM_VERSION} configuration: ")
message("  Triplet: ${CHARM_PLATFORM}")
message("  Network: ${NETWORK}")
message("  C compiler: ${CMAKE_C_COMPILER} [${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}]")
message("  C++ compiler: ${CMAKE_CXX_COMPILER} [${CMAKE_CXX_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}]")
message("  F90 compiler: ${CMAKE_Fortran_COMPILER} [${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}]")
message("  Enabled options:${opts_enabled}")
message("==============================" )

# Make symlinks
add_custom_target(create_symlinks ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/bin ${CMAKE_SOURCE_DIR}/bin
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/lib ${CMAKE_SOURCE_DIR}/lib
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/include ${CMAKE_SOURCE_DIR}/include
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/include ${CMAKE_SOURCE_DIR}/tmp

                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/include ${CMAKE_BINARY_DIR}/tmp
                 )

# Create .vdir/.gdir/VERSION files
file(WRITE ${CMAKE_BINARY_DIR}/include/.gdir   "${NETWORK}\n")
file(WRITE ${CMAKE_BINARY_DIR}/include/.vdir   "${CHARM_PLATFORM}\n")
file(WRITE ${CMAKE_BINARY_DIR}/include/VERSION "${CHARM_VERSION}\n")


# Installation rules
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin DESTINATION . PATTERN * PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE)
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib     DESTINATION .)
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib_so  DESTINATION .)
install(DIRECTORY ${CMAKE_BINARY_DIR}/include DESTINATION .)
