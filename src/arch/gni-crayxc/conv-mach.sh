#  Cray XC build.  October 2016.
#
#  BUILD CHARM++ on the CRAY XC:
#       COMMON FOR ALL COMPILERS
#  ======================================================
#  #  Do not set this env variable when building charm++. This will cause charm++ to fail.
#  #     setenv CRAYPE_LINK_TYPE dynamic
#
#  #  Use 8M hugepages unless explicitly building with charm++'s regularpages option.
#  module load craype-hugepages8M
#
#  #  Then load a compiler's PrgEnv module.
#  #  Do not add 'craycc', 'icc', or 'gcc' to your build options or this will fail to build.
#
#  CRAY COMPILER (CCE) BUILD
#  ========================================
#  module load PrgEnv-cray         # typically default
#  module swap cce cce/8.5.4       # cce/8.5.4 or later required
#
#  INTEL BUILD
#  ================================
#  module swap PrgEnv-cray PrgEnv-intel
#
#  GCC BUILD
#  ================================
#  module swap PrgEnv-cray PrgEnv-gnu
#
#  # Build command is the same regardless of compiler environment:
#
#  # uGNI build
#  ./build charm++ gni-crayxc smp --with-production

GNI_CRAYXC=1

. common-craype.sh
