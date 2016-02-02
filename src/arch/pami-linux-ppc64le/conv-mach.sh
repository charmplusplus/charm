
PAMI_INC=/opt/ibmhpc/pecurrent/ppe.pami/include
PAMI_LIB=/opt/ibmhpc/pecurrent/ppe.pami/gnu/lib64/pami64

CXX=xlC_r
CC=xlc_r

CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C="$CC -E"
CMK_CC="$CC "
CMK_CXX="$CXX "
CMK_CXXPP="$CXX -E "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

CMK_C_OPTIMIZE='-O3 -Q -g'
CMK_CXX_OPTIMIZE='-O3 -Q -g'

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_SYSINC="-I $PAMI_INC"
#CMK_SYSLIBS="-L $PAMI_LIB -L /usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lnuma -lstdc++ -lc -ldl -lrt -lpthread"
CMK_SYSLIBS="-L $PAMI_LIB -L /usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lstdc++ -lc -ldl -lrt -lpthread"

CMK_NATIVE_LIBS=''
CMK_NATIVE_CC="$CC -q64"
CMK_NATIVE_LD="$CC -q64"
CMK_NATIVE_CXX="$CXX -q64"
CMK_NATIVE_LDXX="$CXX -q64"

# fortran compiler
CMK_CF77="xlf_r -q64 -fPIC "
CMK_CF90="xlf90_r -q64 -fPIC -qsuffix=f=f90"
CMK_CF90_FIXED="xlf90_r -q64 -fPIC"

CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_MODINC="-p"
CMK_F90_USE_MODDIR=""

F90DIR=`which ifort 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
if test -x "$F90DIR"
then
  F90LIBDIR="`dirname $F90DIR`/../lib"
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
CMK_F77LIBS="$CMK_F90LIBS"
