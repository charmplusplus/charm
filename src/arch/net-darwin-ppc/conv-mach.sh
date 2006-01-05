CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="cc -E"
CMK_CC="cc -Wno-long-double -fPIC -dynamic -fno-common "
CMK_CXX="c++ -Wno-long-double -fPIC -dynamic -fno-common "
CMK_CXXPP="c++ -x c++ -E "
CMK_LDXX="$CMK_CXX -multiply_defined suppress "
CMK_XIOPTS=""
CMK_QT="generic-light"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"

# Assumes IBM xlf90 compiler:
CMK_CF77="f77 -qnocommon -qextname -qthreaded "
CMK_CF90="f90 -qnocommon -qextname -qthreaded "
CMK_CF90_FIXED="xlf90 -qnocommon -qextname -qthreaded -qsuffix=f=f"
CMK_F90LIBS="-L/opt/ibmcmp/xlf/8.1/lib -lxlf90 -lxlopt -lxl -lxlfmath"
CMK_F77LIBS=$CMK_F90LIBS
CMK_MOD_EXT="mod"

# setting for shared lib
# need -lstdc++ for c++ reference, and it needs to be put at very last 
# of command line.
# Mac environment varaible
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.3
CMK_SHARED_SUF="dylib"
#CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup -flat_namespace "
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lstdc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
