#CMK_DEFS="-I/opt/xt-mpt/1.5.47/mpich2-64/T/include "
#CMK_LD_DEFS="-lrca "

PGCC=`CC -V 2>/dev/null | grep pgCC`

CMK_INC="-I/opt/cray/mpt/5.2.0/xt/gemini/sma/include/"
CMK_LD_PATH="-L/opt/cray/mpt/5.2.0/xt/gemini/sma/lib64/"

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="cc -E $CMK_DEFS $CMK_INC"
CMK_CXXPP="CC -E $CMK_DEFS $CMK_INC"
CMK_CC="cc $CMK_DEFS $CMK_INC"
CMK_CXX="CC  $CMK_DEFS $CMK_INC"
CMK_LD="$CMK_CC $CMK_LD_DEFS $CMK_LD_PATH"
CMK_LDXX="$CMK_CXX $CMK_LD_DEFS $CMK_LD_PATH"
CMK_LIBS="-lckqt -lsma -lrca"
CMK_QT='generic64-light'

CMK_RANLIB="ranlib"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

# compiler for compiling sequential programs
if test -n "$PGCC"
then
CMK_CC="$CMK_CC -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CXX="$CMK_CXX -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
# gcc is needed for building QT
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="pgCC -fPIC --no_using_std "
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


# for F90 compiler
CMK_CF77="ftn "
CMK_CF90="ftn "
CMK_F90LIBS=""
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_MOD_EXT="mod"

CMK_NO_BUILD_SHARED=true

