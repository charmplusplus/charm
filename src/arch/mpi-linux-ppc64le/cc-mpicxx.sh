
# user enviorn var: MPICXX and MPICC
# or, use the definition in file $CHARMINC/MPIOPTS
MPICXX_DEF=mpicxx
MPICC_DEF=mpicc

test "$MPICXX" != "$MPICXX_DEF" && /bin/rm -f $CHARMINC/MPIOPTS
if test ! -f "$CHARMINC/MPIOPTS"
then
  echo MPICXX_DEF=$MPICXX > $CHARMINC/MPIOPTS
  echo MPICC_DEF=$MPICC >> $CHARMINC/MPIOPTS
  chmod +x $CHARMINC/MPIOPTS
fi

CMK_REAL_COMPILER=`$MPICXX -show 2>/dev/null | cut -d' ' -f1 `
case "${CMK_REAL_COMPILER##*/}" in
gcc|g++|gcc-*|g++-*) CMK_AMD64="-fPIC" ;;
pgCC|pgc++|nvc++) CMK_AMD64="-DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std " ;;
esac

CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="$MPICC -E"
CMK_CC="$MPICC $CMK_AMD64 "
CMK_CXX="$MPICXX $CMK_AMD64 "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

# fortran compiler
# for Intel Fortran compiler 8.0 and higher which is renamed to ifort from ifc
# does not work for ifc 7.0
CMK_CF77="mpif77 -auto -fPIC "
CMK_CF90="mpif90 -auto -fPIC "
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
F90DIR=`command -v ifort 2> /dev/null`
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

CMK_COMPILER='mpicc'
