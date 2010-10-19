CMK_64bit=" -m64 "
CMK_PIC="-fpic"

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="gcc -E"
CMK_CC="gcc $CMK_64bit "
CMK_CXX="g++ $CMK_64bit "
CMK_CXXPP="$CMK_CXX -x c++ -E "
CMK_RANLIB="ranlib"
CMK_LIBS="-lckqt"
#CMK_LD="$CMK_CC -Wl,--allow-multiple-definition "
#CMK_LDXX="$CMK_CXX -Wl,--allow-multiple-definition "
CMK_LD="$CMK_CC $CMK_64bit"
CMK_LDXX="$CMK_CXX $CMK_64bit"
CMK_LD_SHARED="-shared"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_XIOPTS=""
CMK_QT="generic64-light"

# fortran compiler Absoft or gnu f95
CMK_CF77="g77 $CMK_64bit"
CMK_F77LIBS="-lg2c "
CMK_CF90=`which xlf90_r 2>/dev/null`
if test -n "$CMK_CF90"
then
# xlf
  bindir=`dirname $CMK_CF90`
  libdir="$bindir/../lib64"
  CMK_64bit=" -q64 "
  CMK_CF90="$CMK_CF90 $CMK_64bit -qpic=large -qthreaded -qlanglvl=90std -qwarn64 -qspill=32648 -qsuppress=1513-029:1518-012:1518-059 -qsuffix=f=f90:cpp=F90 "
  CMK_CF90_FIXED="$CMK_CF90 $CMK_64bit -qsuffix=f=f:cpp=F -qfixed=132 "
  CMK_F90LIBS="-L/opt/ibmcmp/xlf/11.1/bin/../../../xlsmp/1.7/lib64 -L$libdir -lxl -lxlf90 -lxlfmath -lxlopt -lxlsmp"
  CMK_F90_USE_MODDIR=1
  CMK_F90_MODINC="-I"
else
# gnu f95
  CMK_CF90=`which f95 2>/dev/null`
  if test -n "$CMK_CF90"
  then
    CMK_FPP="/lib/cpp -P -CC"
    CMK_CF90="$CMK_CF90 -m64 -fpic -fautomatic -fdollar-ok $CMK_64bit"
    CMK_CF90_FIXED="$CMK_CF90 -m64 -ffixed-form $CMK_64bit"
    CMK_F90LIBS="-lgfortran "
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC="-I"
  fi
fi
