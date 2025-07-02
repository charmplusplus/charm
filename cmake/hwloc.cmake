# Support for building hwloc

# Hardware Locality (hwloc)
if(BUILD_SHARED)
    set(hwloc_shared yes)
else()
    set(hwloc_shared no)
endif()

# Used to determine arguments to pass to hwloc's configure
set(building_blocks_command "${CMAKE_BINARY_DIR}/bin/charmc -print-building-blocks ${OPTS}")

set(hwloc_dir ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc)
include(ExternalProject)
ExternalProject_Add(hwloc
    URL ${CMAKE_SOURCE_DIR}/contrib/hwloc
    CONFIGURE_COMMAND
      bash "-c" "eval $(${building_blocks_command}) && \
        ${hwloc_dir}/configure \
        --disable-cairo \
        --disable-cuda \
        --disable-gl \
        --disable-levelzero \
        --disable-libxml2 \
        --disable-nvml \
        --disable-opencl \
        --disable-pci \
        --disable-rsmi \
        --disable-libudev \
        --disable-visibility \
        --enable-embedded-mode \
        --enable-shared=${hwloc_shared} \
        --enable-static \
        --with-hwloc-symbol-prefix=cmi_ \
        --without-x \
        HWLOC_FLAGS=\"$CHARM_CC_FLAGS\" \
        CFLAGS=\"$CHARM_CC_FLAGS\" \
        CXXFLAGS=\"$CHARM_CXX_FLAGS\" \
        CC=\"$CHARM_CC\" \
        CXX=\"$CHARM_CXX\" \
        CC_FOR_BUILD=\"$CHARM_CC\" \
        CPP= \
        CXXCPP= \
        CPPFLAGS= \
        LDFLAGS= \
        LIBS= \
        > /dev/null"
    BUILD_COMMAND $(MAKE) V=$(VERBOSE) AUTOCONF=: AUTOHEADER=: AUTOMAKE=: ACLOCAL=:
    INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc/include/hwloc.h ${CMAKE_BINARY_DIR}/include/
    COMMAND cp -LRf ${CMAKE_SOURCE_DIR}/contrib/hwloc/include/hwloc ${CMAKE_BINARY_DIR}/include/
    COMMAND cp -f ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/include/hwloc/autogen/config.h ${CMAKE_BINARY_DIR}/include/hwloc/autogen/
)

if(CMK_WINDOWS)
    set(obj_suf ".obj")
else()
    set(obj_suf ".o")
endif()

# These objects are used to embed hwloc in libconverse
set(hwloc-objects
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/base64${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/bind${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/bitmap${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/components${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/cpukinds${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/diff${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/distances${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/memattrs${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/misc${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/pci-common${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/shmem${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-noos${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-synthetic${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-xml-nolibxml${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-xml${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology${obj_suf}
  ${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/traversal${obj_suf}
)

if(CMK_MACOSX)
  set(hwloc-objects "${hwloc-objects};${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-darwin.o")
elseif(CMK_WINDOWS)
  set(hwloc-objects "${hwloc-objects};${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-windows.obj")
else() # Linux
  set(hwloc-objects "${hwloc-objects};${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-linux.o")
  set(hwloc-objects "${hwloc-objects};${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-hardwired.o")
endif()

if((CHARM_CPU STREQUAL "i386" OR CHARM_CPU STREQUAL "x86_64") AND NOT CMK_WINDOWS)
  set(hwloc-objects "${hwloc-objects};${CMAKE_BINARY_DIR}/hwloc-prefix/src/hwloc-build/hwloc/topology-x86.o")
endif()

set_source_files_properties(
  ${hwloc-objects}
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
)
