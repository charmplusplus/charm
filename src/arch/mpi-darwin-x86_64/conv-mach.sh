CMK_MACOSX=1

# user enviorn var: MPICXX and MPICC
# or, use the definition in file $CHARMINC/MPIOPTS
if test -x "$CHARMINC/MPIOPTS"
then
  . $CHARMINC/MPIOPTS
else
  MPICXX_DEF=mpicxx
  MPICC_DEF=mpicc
fi

test -z "$MPICXX" && MPICXX=$MPICXX_DEF
test -z "$MPICC" && MPICC=$MPICC_DEF
test "$MPICXX" != "$MPICXX_DEF" && /bin/rm -f $CHARMINC/MPIOPTS
if test ! -f "$CHARMINC/MPIOPTS"
then
  echo MPICXX_DEF=$MPICXX > $CHARMINC/MPIOPTS
  echo MPICC_DEF=$MPICC >> $CHARMINC/MPIOPTS
  chmod +x $CHARMINC/MPIOPTS
fi

CMK_REAL_COMPILER=`$MPICXX -show 2>/dev/null | cut -d' ' -f1 `

CMK_AMD64="-m64 -dynamic -fPIC -fno-common -mmacosx-version-min=10.7"

CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="$MPICC -E -mmacosx-version-min=10.7"
CMK_CC="$MPICC $CMK_AMD64 "
CMK_CXX="$MPICXX $CMK_AMD64 "
CMK_CXXPP="$MPICXX -E $CMK_AMD64 "

CMK_XIOPTS=""
CMK_QT="generic64-light"
CMK_LIBS="-lckqt $CMK_SYSLIBS "
CMK_RANLIB="ranlib"

CMK_NATIVE_CC="clang $CMK_GCC64 "
CMK_NATIVE_LD="clang -Wl,-no_pie $CMK_GCC64 "
CMK_NATIVE_CXX="clang++ $CMK_GCC64 -stdlib=libc++ "
CMK_NATIVE_LDXX="clang++ -Wl,-no_pie $CMK_GCC64 -stdlib=libc++ "
CMK_NATIVE_LIBS=""

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
    . $CHARMINC/conv-mach-gfortran.sh
else
    CMK_CF77="g77 "
    CMK_CF90="f90 "
    CMK_CF90_FIXED="$CMK_CF90 -W132 "
    CMK_F90LIBS="-lf90math -lfio -lU77 -lf77math "
    CMK_F77LIBS="-lg2c "
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC="-p"
fi

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last 
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.5
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
