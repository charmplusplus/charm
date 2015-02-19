#CMK_DEFS="-I/opt/xt-mpt/1.5.47/mpich2-64/T/include "
#CMK_LD_DEFS="-lrca "

CMK_BUILD_CRAY=1

PGCC=`CC -V 2>&1 | grep pgCC`
ICPC=`CC -V 2>&1 | grep Intel`
GNU=`CC -V 2>&1 | grep 'g++'`

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
CMK_CXX="$CMK_CXX -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
# gcc is needed for building QT
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_CXX="pgCC -fPIC --no_using_std "
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
CMK_QT="generic64-light"

# for F90 compiler
CMK_CF77="ftn "
CMK_CF90="ftn "
if test -n "$GNU"
then
    CMK_CF77="$CMK_CF77 -ffree-line-length-none"
    CMK_CF90="$CMK_CF90 -ffree-line-length-none"
fi
CMK_F90LIBS=""
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_MOD_EXT="mod"

CMK_NO_BUILD_SHARED=true

