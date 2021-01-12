include(cmake/detect-features-c.cmake)
include(cmake/detect-features-cxx.cmake)
include(cmake/detect-features-fortran.cmake)

# Programs
find_program(SYNC sync)
if(SYNC)
    set(CMK_HAS_SYNC 1)
endif()


set(CMK_MACHINE_NAME \"${CHARM_PLATFORM}\")

set(CMK_CCS_AVAILABLE 1)
if(${NETWORK} STREQUAL "pami" OR ${NETWORK} STREQUAL "pamilrts")
  set(CMK_CCS_AVAILABLE 0)
endif()

set(CMK_NO_PARTITIONS 0)
if(${NETWORK} STREQUAL "netlrts" OR ${NETWORK} STREQUAL "multicore" OR ${NETWORK} STREQUAL "pami")
  set(CMK_NO_PARTITIONS 1)
endif()

set(CMK_HAS_OPENMP ${OPENMP_FOUND})

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # TODO: Apple clang needs external library for OpenMP support, disable for now.
  set(CMK_HAS_OPENMP 0)
endif()

# CMA
set(CMK_USE_CMA ${CMK_HAS_CMA})
if(NETWORK STREQUAL "multicore" OR NETWORK MATCHES "bluegeneq")
  set(CMK_USE_CMA 0)
endif()


# Misc. linker flags (mostly for Charm4py)
include(CheckCCompilerFlag)
set(CMAKE_REQUIRED_FLAGS "-Wl,--no-as-needed")
check_c_compiler_flag("" CXX_NO_AS_NEEDED)
if(CXX_NO_AS_NEEDED)
  set(CXX_NO_AS_NEEDED "-Wl,--no-as-needed")
else()
  set(CXX_NO_AS_NEEDED "")
endif()

set(CMAKE_REQUIRED_FLAGS "-Wl,--whole-archive -Wl,--no-whole-archive")
check_c_compiler_flag("" LDXX_WHOLE_ARCHIVE)
if(LDXX_WHOLE_ARCHIVE)
  set(LDXX_WHOLE_ARCHIVE_PRE "-Wl,--whole-archive") # Flags for Linux ld
  set(LDXX_WHOLE_ARCHIVE_POST "-Wl,--no-whole-archive")
else()
  set(LDXX_WHOLE_ARCHIVE_PRE " ")
  set(LDXX_WHOLE_ARCHIVE_POST " ")
endif()

if(${LDXX_WHOLE_ARCHIVE_PRE} STREQUAL " ")
    set(CMAKE_REQUIRED_FLAGS "-Wl,-all_load") # Flag for MacOS ld
    check_c_compiler_flag("" LDXX_ALL_LOAD)
    if(LDXX_ALL_LOAD)
      set(LDXX_WHOLE_ARCHIVE_PRE "-Wl,-all_load")
    else()
      set(LDXX_WHOLE_ARCHIVE_PRE " ")
    endif()
    set(LDXX_WHOLE_ARCHIVE_POST " ")
endif()

set(CMAKE_REQUIRED_FLAGS "")

# Support for fsglobals/pipglobals
if(CMK_WINDOWS)
  set(CMK_SUPPORTS_FSGLOBALS 1)
elseif(CMK_HAS_DLOPEN AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMK_SUPPORTS_FSGLOBALS 1)
elseif(CMK_HAS_DLOPEN AND (${CMK_HAS_READLINK} OR ${CMK_HAS_REALPATH}))
  set(CMK_SUPPORTS_FSGLOBALS 1)
else()
  set(CMK_SUPPORTS_FSGLOBALS 0)
endif()

if(CMK_WINDOWS)
  set(CMK_CAN_OPEN_SHARED_OBJECTS_DYNAMICALLY 1)
else()
  set(CMK_CAN_OPEN_SHARED_OBJECTS_DYNAMICALLY ${CMK_HAS_DLOPEN})
endif()

if(CMK_HAS_DLMOPEN AND CMK_CAN_OPEN_SHARED_OBJECTS_DYNAMICALLY)
  set(CMK_SUPPORTS_PIPGLOBALS 1)
else()
  set(CMK_SUPPORTS_PIPGLOBALS 0)
endif()

# Misc. flags
set(CMK_LBID_64BIT 1)
set(CMK_CKSECTIONINFO_STL 1)

#FIXME: add CMK_CRAY_MAXNID


# Create conv-autoconfig.h by iterating over all variable names and #defining them.
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)

list(REMOVE_ITEM _variableNames CMK_USE_CMA)

set(optfile ${CMAKE_BINARY_DIR}/include/conv-autoconfig.h)
file(REMOVE ${optfile})

foreach (v ${_variableNames})
    if(("${v}" MATCHES "^CMK_"  OR "${v}" MATCHES "^SIZEOF_" OR "${v}" MATCHES "^CHARM_" OR "${v}" MATCHES "^QLOGIC$") AND NOT "${v}" MATCHES "_CODE$")
        if("${${v}}" STREQUAL "" OR "${${v}}" STREQUAL "FALSE")
            set(${v} 0)
        elseif("${${v}}" STREQUAL "TRUE")
            set(${v} 1)
        endif()
        file(APPEND ${optfile} "#define ${v} ${${v}}\n" )
    elseif("${v}" MATCHES "^HAVE_")
        if("${${v}}" STREQUAL "" OR "${${v}}" STREQUAL "FALSE")
            set(${v} 0)
            file(APPEND ${optfile} "/* #define ${v} ${${v}} */\n" )
        elseif("${${v}}" STREQUAL "TRUE")
            set(${v} 1)
            file(APPEND ${optfile} "#define ${v} ${${v}}\n" )
        endif()
    endif()
endforeach()
