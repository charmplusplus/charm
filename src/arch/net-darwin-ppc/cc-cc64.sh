COMMENTS="gcc 64bit for MacOSX Tiger"

CMK_MACOSX64=1
CMK_CC="cc -m64 -fPIC -dynamic -fno-common "
CMK_CXX="c++ -m64 -fPIC -dynamic -fno-common "
CMK_CXXPP="c++ -m64 -x c++ -E "
CMK_LD="$CMK_CC -multiply_defined suppress "
CMK_LDXX="$CMK_CXX -multiply_defined suppress "
CMK_QT="generic64-light"

# Assumes IBM xlf90 compiler:
CMK_CF77="f77 -qnocommon -qextname -qthreaded "
CMK_CF90="f90 -qnocommon -qextname -qthreaded "
CMK_CF90_FIXED="xlf90 -qnocommon -qextname -qthreaded -qsuffix=f=f"
CMK_F90LIBS="-L/opt/ibmcmp/xlf/8.1/lib -lxlf90 -lxlopt -lxl -lxlfmath"

# loading all members of archives on the command line
CMK_LD_SHARED="$CMK_LD_SHARED -all_load"
