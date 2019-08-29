CMK_CPP_C='icc -E '
CMK_CC="icc -fpic "
CMK_CXX="icpc -fpic "

CMK_LD="icc -shared-intel "
CMK_LDXX="icpc -shared-intel "

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CPPFLAGS="$CPPFLAGS -fpic "
LDFLAGS="$LDFLAGS -shared-intel "

CMK_CF77="ifort -auto -fPIC "
CMK_CF90="ifort -auto -fPIC "
#CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
#FOR 64 bit machine
CMK_CF90_FIXED="$CMK_CF90 -164 -FI "
F90DIR=`which ifort 2> /dev/null`
if test -x "$F90DIR"
then
  MYDIR="$PWD"
  cd `dirname "$F90DIR"`
  if test -L 'ifort'
  then
    F90DIR=`readlink ifort`
    cd `dirname "$F90DIR"`
  fi
  F90DIR=`pwd -P`
  cd "$MYDIR"

  Minor=`basename $F90DIR`
  F90LIBDIR="$F90DIR/../lib/$Minor"
  if ! test -x "$F90LIBDIR"
  then
    F90LIBDIR="$F90DIR/../lib"
    if ! test -x "$F90LIBDIR"
    then
      F90LIBDIR="$F90DIR/../../compiler/lib/$Minor"
    fi
    if ! test -x "$F90LIBDIR"
    then
      F90LIBDIR="$F90DIR/../../lib/$Minor"
    fi
    if ! test -x "$F90LIBDIR"
    then
      F90LIBDIR="$F90DIR/../../compiler/lib/${Minor}_lin"
    fi
    if ! test -x "$F90LIBDIR"
    then
      F90LIBDIR="$F90DIR/../../lib/${Minor}_lin"
    fi
  fi
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport -lifcore "
CMK_F77LIBS="$CMK_F90LIBS"

CMK_F90_USE_MODDIR=""

CMK_C_OPENMP="-fopenmp"

CMK_COMPILER='icc'
