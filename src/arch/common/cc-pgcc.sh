
# machine specific recommendation
case `hostname` in
*.ranger.tacc.utexas.edu) CMK_DEFS="$CMK_DEFS -tp barcelona-64" ;;
esac

CMK_CPP_C="pgcc -E "
CMK_CC="pgcc -fPIC -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CC_RELIABLE="gcc "
#CMK_CXX="pgCC --instantiate=used "
CMK_CXX="pgCC -fPIC -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

# compiler for compiling sequential programs
# pgcc can not handle QT right for generic64, so always use gcc
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_LD="$CMK_SEQ_CC "
CMK_SEQ_CXX="pgCC -fPIC --no_using_std "
CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_SEQ_LIBS=""

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

# fortran compiler
CMK_CF77="pgf77 "
CMK_CF90="pgf90 "
CMK_CF90_FIXED="$CMK_CF90 -Mfixed "
f90libdir="."
f90bin=`command -v pgf90 2>/dev/null`
PG_DIR="`dirname $f90bin`/.."
if test -n "$PG_DIR"
then
  f90libdir="$PG_DIR/lib"
fi
CMK_F90LIBS="-L$f90libdir  -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl -lpgftnrtl "
CMK_F90_USE_MODDIR=""

CMK_COMPILER='pgcc'
