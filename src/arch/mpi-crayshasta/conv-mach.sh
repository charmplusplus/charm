#  Cray Shasta build.  October 2021.
#  CRAY COMPILER (CCE) BUILD
#  ========================================
#  module load PrgEnv-cray         # typically default
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
#  # MPI build
#  ./build charm++ mpi-crayshasta smp --with-production

CMK_BUILD_CRAY=1

CMK_CRAY_NOGNI=1

. $CHARMINC/conv-mach-craype.sh
