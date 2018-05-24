. $CHARMINC/cc-gcc.sh

CMK_CPP_C_FLAGS="$CMK_CPP_C_FLAGS -m32"
CMK_CC_FLAGS="$CMK_CC_FLAGS -m32"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -m32"
#CMK_LD="$CMK_CC -Wl,--allow-multiple-definition "
#CMK_LDXX="$CMK_CXX -Wl,--allow-multiple-definition "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "
CMK_LD_FLAGS="$CMK_CC_FLAGS "
CMK_LDXX_FLAGS="$CMK_CXX_FLAGS "
CMK_XIOPTS=''
CMK_QT='i386-gcc'

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
    . $CHARMINC/conv-mach-gfortran.sh
else

# fortran compiler f77 or gnu f95
CMK_CF77=`which g77 2>/dev/null`
if test -z "$CMK_CF77"
then
  CMK_CF77=`which f77 2>/dev/null`
fi
if test -n "$CMK_CF77"
then
  CMK_F77LIBS='-lg2c '
fi

CMK_CF90=`which f90 2>/dev/null`
if test -n "$CMK_CF90"
then
# absoft
  CMK_CF90_FIXED="$CMK_CF90 -W132 "
  CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
  CMK_F90_USE_MODDIR=1
  CMK_F90_MODINC='-p'
else
# gnu f95
  CMK_CF90=`which f95 2>/dev/null`
  if test -n "$CMK_CF90"
  then
    CMK_FPP='cpp -P -CC'
    CMK_CF90="$CMK_CF90 -fpic -fautomatic -fdollar-ok "
    CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "
    CMK_F90LIBS='-lgfortran '
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC='-I'
  fi
fi

fi

CMK_CF77="$CMK_CF77 -m32"
CMK_CF90="$CMK_CF90 -m32"
CMK_CF90_FIXED="$CMK_CF90_FIXED -m32"
