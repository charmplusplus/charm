#CMK_DEFS="-I/opt/xt-mpt/1.5.47/mpich2-64/T/include "
#CMK_LD_DEFS="-lrca "

CMK_BUILD_CRAY=1

PGCC=`CC -V 2>&1 | grep pgCC`
ICPC=`CC -V 2>&1 | grep Intel`

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="cc -E $CMK_DEFS "
CMK_CXXPP="CC -E $CMK_DEFS "
CMK_CC="cc $CMK_DEFS "
CMK_CXX="CC  $CMK_DEFS "
CMK_LD="$CMK_CC $CMK_LD_DEFS"
CMK_LDXX="$CMK_CXX $CMK_LD_DEFS"
# Swap these and set XT[45]_TOPOLOGY in conv-mach.h if doing topo work
# on a Cray XT of known dimensions. See src/util/CrayNid.c for details
#CMK_LIBS="-lckqt -lrca"
CMK_LIBS="-lckqt"

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

# compiler for compiling sequential programs
if test -n "$PGCC"
then
CMK_CC="$CMK_CC -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CXX="$CMK_CXX -DCMK_FIND_FIRST_OF_PREDICATE=1 "
# gcc is needed for building QT
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_CXX="pgCC -fPIC "
elif test -n "$ICPC"
then
CMK_SEQ_CC="icc -fPIC "
CMK_SEQ_CXX="icpc -fPIC "
else
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="g++ -fPIC "
fi
CMK_SEQ_LD="$CMK_SEQ_CC "
CMK_SEQ_LDXX="$CMK_SEQ_CXX "
CMK_SEQ_LIBS=""

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

CMK_RANLIB="ranlib"
CMK_QT="generic64"

# for F90 compiler
CMK_CF77="ftn "
CMK_CF90="ftn "
if test -n "$ICPC"
then
  F90DIR=`which ifort 2> /dev/null`
  if test -h "$F90DIR"
  then
    F90DIR=`readlink $F90DIR`
  fi
  if test -x "$F90DIR"
  then
    F90DIR=`dirname $F90DIR`
    Minor=`basename $F90DIR`
    if test "$Minor" = "intel64"
    then
      F90DIR=`dirname $F90DIR`
      F90LIBDIR="$F90DIR/../lib/$Minor"
    else
      F90LIBDIR="$F90DIR/../lib"
    fi
    F90MAIN="$F90LIBDIR/for_main.o"
  fi
  # for_main.o is important for main() in f90 code
  CMK_F90MAINLIBS="$F90MAIN "
  CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport -lsvml "
  CMK_F77LIBS="$CMK_F90LIBS"
  CMK_F90_USE_MODDIR=""
else
  CMK_F90LIBS=""
  CMK_F90_USE_MODDIR=1
  CMK_F90_MODINC="-I"
  CMK_MOD_EXT="mod"
fi

CMK_NO_BUILD_SHARED=true

