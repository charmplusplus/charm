CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C='mpicc -E'
CMK_CC='mpicc -Wno-long-double -fPIC -dynamic -fno-common '
CMK_CXX='mpicxx -Wno-long-double -fPIC -dynamic -fno-common '
CMK_CXXPP='mpicxx -E '
CMK_LDXX="$CMK_CXX -multiply_defined suppress "
CMK_CF77='f77'
CMK_CF90='f90 -qnocommon -qextname '
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lmpich '
CMK_NATIVE_LIBS=''
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc'
CMK_NATIVE_CXX='g++ '
CMK_NATIVE_LDXX='g++'
CMK_NATIVE_CC='gcc '
CMK_NATIVE_CXX='g++ '
# Assumes IBM xlf90 compiler:
CMK_F90LIBS="-L/opt/ibmcmp/xlf/8.1/lib -lxlf90 -lxlopt -lxl -lxlfmath"
CMK_MOD_EXT="mod"

# setting for shared lib
# need -lstdc++ for c++ reference, and it needs to be put at very last
# of command line.
# Mac environment varaible
export MACOSX_DEPLOYMENT_TARGET=10.3
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup -single_module"
CMK_LD_SHARED_LIBS="-lstdc++"

