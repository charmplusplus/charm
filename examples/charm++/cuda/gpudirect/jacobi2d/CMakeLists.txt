cmake_minimum_required(VERSION 3.16)
project(jacobi2d)

include(${CMAKE_SOURCE_DIR}/../../../../../config.cmk)

set(Kokkos_ROOT ${KOKKOS_DIR})
find_package(Kokkos 4.5 REQUIRED CONFIG)

#set(KokkosKernels_DIR ${KOKKOS_KERNELS_DIR}/lib64/cmake/KokkosKernels)
#find_package(KokkosKernels REQUIRED)

if(Charm_ENABLE_GPU)
    message(STATUS "Building for a GPU backend")
    add_definitions(-DGPU_BACKEND)
    add_definitions(-DCUDA_DIR=\"${CUDA_DIR}\")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_OPTS} ${LD_OPTS}")
else()
    message(STATUS "Building for a CPU backend")
    set(CMAKE_CXX_COMPILER "${CHARMC}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPU_OPTS} ${LD_OPTS}")
endif()

add_definitions(-DKOKKOS_DIR=\"${KOKKOS_DIR}\")

add_custom_command(
    OUTPUT  ${CMAKE_SOURCE_DIR}/jacobi2d.decl.h
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}
    ${CHARMC} ${CMAKE_SOURCE_DIR}/jacobi2d.ci
    DEPENDS ${CMAKE_SOURCE_DIR}/jacobi2d.ci
    COMMENT "Processing jacobi2d ci files"
)

if(Charm_ENABLE_GPU)
    add_library(jacobi2d OBJECT ${CMAKE_SOURCE_DIR}/jacobi2d.C ${CMAKE_SOURCE_DIR}/jacobi2d.decl.h)
    target_include_directories(jacobi2d PRIVATE ${CHARM_DIR}/include)
    target_link_libraries(jacobi2d Kokkos::kokkos)

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/jacobi2d.out"
        COMMAND ${CHARMC} ${GPU_LINK_OPTS} $<TARGET_OBJECTS:jacobi2d> -o ${CMAKE_BINARY_DIR}/jacobi2d.out
        DEPENDS jacobi2d
        COMMENT "Linking charm build against kokkos and cuda"
    )

    add_custom_target(result ALL
        DEPENDS "${CMAKE_BINARY_DIR}/jacobi2d.out"
    )
else()
    add_executable(jacobi2d.out ${CMAKE_SOURCE_DIR}/jacobi2d.C ${CMAKE_SOURCE_DIR}/jacobi2d.decl.h)
    target_include_directories(jacobi2d.out PRIVATE ${CHARM_DIR}/include)
    target_link_libraries(jacobi2d.out Kokkos::kokkos)
endif()
