
cmake_minimum_required(VERSION 3.4)

project(Charm++)

# message(STATUS "testprint: ${CMAKE_CXX_FLAGS}")

set(_CHARM_NETWORK_TYPE "multicore")
set(CHARM_NETWORK_TYPE "${_CHARM_NETWORK_TYPE}" CACHE STRING "Target network layer. ex: multicore/netlrts/pamilrts/verbs/mpi (Only effective before first configure. See CHARM_PLATFORM.)")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_CHARM_OS "darwin")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_CHARM_OS "win")
else()
    set(_CHARM_OS "linux")
endif()
set(CHARM_OS "${_CHARM_OS}" CACHE STRING "Target operating system. ex: linux/darwin/win (Only effective before first configure. See CHARM_PLATFORM.)")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_CHARM_CPU "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "powerpc")
    set(_CHARM_CPU "ppc")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc")
    set(_CHARM_CPU "ppc")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm7")
    set(_CHARM_CPU "arm7")
else()
    set(_CHARM_CPU "")
endif()
set(CHARM_CPU "${_CHARM_CPU}" CACHE STRING "Target CPU architecture. Can be blank. ex: x86_64/ppc/arm7 (Only effective before first configure. See CHARM_PLATFORM.)")

set(_CHARM_PLATFORM "${CHARM_NETWORK_TYPE}-${CHARM_OS}")
if(NOT "${CHARM_CPU}" STREQUAL "")
    set(_CHARM_PLATFORM "${_CHARM_PLATFORM}-${CHARM_CPU}")
endif()
set(CHARM_PLATFORM "${_CHARM_PLATFORM}" CACHE STRING "Target platform tuplet. (Constructed from CHARM_NETWORK_TYPE, CHARM_OS, and CHARM_CPU if not specified.)")

set(_CHARM_BUILD_OPTIONS "")

option(CHARM_SMP "Enable shared-memory communication within a single multi-processor machine instead of message passing.")
if (CHARM_SMP)
    list(APPEND _CHARM_BUILD_OPTIONS "smp")
endif()

set(_CHARM_COMPILER "")
set(CHARM_COMPILER "${_CHARM_COMPILER}" CACHE STRING "Compiler name as understood by the Charm++ build script. ex: gcc/clang/icc/mpicxx/pgcc/xlc")

set (_CHARM_BUILD_FLAGS "")
set (_CHARM_TARGET_OPTS "")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND _CHARM_BUILD_FLAGS "--with-production")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND _CHARM_BUILD_FLAGS "--enable-charmdebug")
    list(APPEND _CHARM_BUILD_FLAGS "--enable-error-checking")
    list(APPEND _CHARM_TARGET_OPTS "-g")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    list(APPEND _CHARM_BUILD_FLAGS "--with-production")
    list(APPEND _CHARM_BUILD_FLAGS "--enable-charmdebug")
    list(APPEND _CHARM_BUILD_FLAGS "--enable-error-checking")
    list(APPEND _CHARM_TARGET_OPTS "-g")
endif()

include(ExternalProject)
ExternalProject_Add(
    Charm++
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build charm++ ${CHARM_PLATFORM} ${_CHARM_BUILD_OPTIONS} ${CHARM_COMPILER} --only-configure ${_CHARM_BUILD_FLAGS} ${_CHARM_TARGET_OPTS} ${CMAKE_CXX_FLAGS}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -C tmp charm++ OPTS="${_CHARM_TARGET_OPTS} ${CMAKE_CXX_FLAGS}"
    # INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} -C tmp install
    # INSTALL_COMMAND true
    INSTALL_COMMAND ""
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

ExternalProject_Get_Property(Charm++ BINARY_DIR)

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(CHARMC_EXECUTABLE ${BINARY_DIR}/bin/charmc PARENT_SCOPE)
    set(CMAKE_CHARM_COMPILER ${BINARY_DIR}/bin/charmc PARENT_SCOPE)
endif()
