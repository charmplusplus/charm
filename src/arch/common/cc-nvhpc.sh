
CMK_CPP_C="nvc -E "
CMK_CC="nvc -fPIC -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CC_RELIABLE="gcc "
CMK_CXX="nvc++ -fPIC -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

# compiler for compiling sequential programs
# nvc can not handle QT right for generic64, so always use gcc
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_LD="$CMK_SEQ_CC "
CMK_SEQ_CXX="nvc++ -fPIC --no_using_std "
CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_SEQ_LIBS=""

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

# fortran compiler
CMK_CF77="nvfortran "
CMK_CF90="nvfortran "
CMK_CF90_FIXED="$CMK_CF90 -Mfixed "
f90libdir="."
f90bin=`command -v nvfortran 2>/dev/null`
NVHPC_DIR="`dirname $f90bin`/.."
if test -n "$NVHPC_DIR"
then
  f90libdir="$NVHPC_DIR/lib"
fi
CMK_F90MAINLIBS="$f90libdir/f90main.o"
CMK_F90LIBS="-L$f90libdir -lnvf"
CMK_F90_USE_MODDIR=""

CMK_COMPILER='nvhpc'
