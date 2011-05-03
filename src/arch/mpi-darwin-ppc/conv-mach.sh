# preprocess mpicxx which include mpich before ampi
# this default conv-mach does not work with linking ampi

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="mpicc -E"
OPTS_CC="$OPTS_CC -dynamic -fno-common "
CMK_CC="mpicc -fPIC "
#CMK_CXX="/private/automount/home/gzheng/csar/genx/build_charm_mpi/mpicxx -fPIC -dynamic -fno-common "
OPTS_CXX="$OPTS_CXX -dynamic -fno-common "
CMK_CXX="mpicxx -fPIC "
CMK_CXXPP="mpicxx -E "
OPTS_LD="$OPTS_LD -multiply_defined suppress -flat_namespace "
CMK_LD="$CMK_CC "
OPTS_LDXX="$OPTS_LDXX -multiply_defined suppress -flat_namespace "
CMK_LDXX="$CMK_CXX "
CMK_LIBS="-lckqt -lmpich -lpmpich"

CMK_QT="generic-light"

# Assumes IBM xlf90 compiler:
CMK_CF77="f77 -qnocommon -qextname -qthreaded "
CMK_CF90="f90 -qnocommon -qextname -qthreaded "
CMK_CF90_FIXED="xlf90 -qnocommon -qextname -qthreaded -qsuffix=f=f"
CMK_F90LIBS="-L/opt/ibmcmp/xlf/8.1/lib -lxlf90 -lxlopt -lxl -lxlfmath"

# native compilers
CMK_NATIVE_LIBS=""
CMK_NATIVE_CC="gcc"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_CXX="g++"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"

# setting for shared lib
# need -lstdc++ for c++ reference, and it needs to be put at very last
# of command line.
# need 10.3 in this Mac environment varaible
export MACOSX_DEPLOYMENT_TARGET=10.3
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -flat_namespace -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lstdc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true

CMK_NO_DL=true
CMK_RANLIB="ranlib -s"
