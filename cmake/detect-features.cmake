include(cmake/detect-features-c.cmake)
include(cmake/detect-features-cxx.cmake)
include(cmake/detect-features-fortran.cmake)

# Programs
find_program(SYNC sync)
if(SYNC)
    set(CMK_HAS_SYNC 1)
endif()


set(CMK_MACHINE_NAME \"${CHARM_PLATFORM}\")
set(CHARM_VERSION ${PROJECT_VERSION})
set(CMK_CCS_AVAILABLE 1)
if(${NETWORK} STREQUAL "uth" OR ${NETWORK} STREQUAL "pami" OR ${NETWORK} STREQUAL "pamilrts")
  set(CMK_CCS_AVAILABLE 0)
endif()

set(CMK_NO_PARTITIONS 0)
if(${NETWORK} STREQUAL "netlrts" OR ${NETWORK} STREQUAL "multicore" OR ${NETWORK} STREQUAL "uth" OR ${NETWORK} STREQUAL "pami" OR ${NETWORK} STREQUAL "shmem" OR ${NETWORK} STREQUAL "sim")
  set(CMK_NO_PARTITIONS 1)
endif()

set(CMK_HAS_OPENMP ${OPENMP_FOUND})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
  # TODO: Apple clang needs external library for OpenMP support, disable for now.
  set(CMK_HAS_OPENMP 0)
endif()


# Create conv-autoconfig.h by iterating over all variable names and #defining them.
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)

set(optfile ${CMAKE_BINARY_DIR}/include/conv-autoconfig.h)
file(REMOVE ${optfile})

foreach (v ${_variableNames})
    if(("${v}" MATCHES "^CMK_"  OR "${v}" MATCHES "^SIZEOF_" OR "${v}" MATCHES "^CHARM_") AND NOT "${v}" MATCHES "_CODE$")
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
