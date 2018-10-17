PAMI_INC=${MPI_ROOT}/pami_devel/include
PAMI_LIB=${MPI_ROOT}/lib/pami_port

LIBCOLL_INC=${MPI_ROOT}/pami_devel/include
LIBCOLL_LIB=${MPI_ROOT}/lib/pami_port

CXX=xlC_r
CC=xlc_r

CMK_CPP_CHARM='cpp -P'
CMK_CPP_C="$CC"
CMK_CC="$CC "
CMK_CXX="$CXX "
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX "

CMK_CPP_C_FLAGS="-E"

CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_SYSINC="-I $PAMI_INC -I $LIBCOLL_INC"
#CMK_SYSLIBS="-L $PAMI_LIB -L /usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lnuma -lstdc++ -lc -ldl -lrt -lpthread"
CMK_SYSLIBS="-L $PAMI_LIB -L $LIBCOLL_LIB -lcollectives -L /usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lstdc++ -lc -ldl -lrt -lpthread"

CMK_NATIVE_LIBS=''
CMK_NATIVE_DEFS='-q64'

# fortran compiler
CMK_CF77='xlf_r -q64 -fPIC '
CMK_CF90='xlf90_r -q64 -fPIC -qsuffix=f=f90'
CMK_CF90_FIXED='xlf90_r -q64 -fPIC'

CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT='mod'
CMK_F90_MODINC='-p'
CMK_F90_USE_MODDIR=''

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
  fi
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport -lifcore "
CMK_F77LIBS="$CMK_F90LIBS"

CMK_COMPILER='xlc'
CMK_NO_PARTITIONS="1"
