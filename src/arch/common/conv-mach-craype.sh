
PGCC=`CC -V 2>&1 | grep pgCC`
ICPC=`CC -V 2>&1 | grep Intel`
GNU=`CC -V 2>&1 | grep 'g++'`
CCE=`CC -V 2>&1 | grep 'Cray'`

CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="cc"
CMK_CC="cc "
CMK_CXX="CC "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

CMK_CPP_C_FLAGS="-E"

CMK_LIBS="-lckqt $CRAY_PMI_POST_LINK_OPTS $CRAY_UGNI_POST_LINK_OPTS -dynamic -lugni -lpmi"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"


CMK_QT="generic64-light"

# compiler for compiling sequential programs
if test -n "$PGCC"
then
CMK_CC_FLAGS="$CMK_CC_FLAGS -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
# gcc is needed for building QT
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_CXX="pgCC -fPIC --no_using_std "
CMK_COMPILER='pgcc'
elif test -n "$CCE"
then
CMK_CXX_OPTIMIZE=" -hipa4"   # For improved C++ performance
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="g++ -fPIC "
CMK_COMPILER='craycc'
elif test -n "$ICPC"
then
CMK_SEQ_CC="icc -fPIC "
CMK_SEQ_CXX="icpc -fPIC "
CMK_COMPILER='icc'
else   # gcc
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="g++ -fPIC "
CMK_COMPILER='gcc'
fi
CMK_SEQ_LD="$CMK_SEQ_CC "
CMK_SEQ_LDXX="$CMK_SEQ_CXX "
CMK_SEQ_LIBS=""

CMK_SEQ_CC_FLAGS=' '
CMK_SEQ_CXX_FLAGS=' '
CMK_SEQ_LD_FLAGS=' '
CMK_SEQ_LDXX_FLAGS=' '

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

CMK_NATIVE_CC_FLAGS=' '
CMK_NATIVE_CXX_FLAGS=' '
CMK_NATIVE_LD_FLAGS=' '
CMK_NATIVE_LDXX_FLAGS=' '

CMK_RANLIB="ranlib"

# for F90 compiler
if test -n "$ICPC"
then
CMK_CF77="ftn -auto "
CMK_CF90="ftn -auto "
CMK_F90LIBS="-lifcore -lifport -lifcore "
else
CMK_CF77="ftn "
CMK_CF90="ftn "
CMK_F90LIBS=""
fi

if test -n "$GNU"
then
    CMK_CF77="$CMK_CF77 -ffree-line-length-none"
    CMK_CF90="$CMK_CF90 -ffree-line-length-none"
    CMK_F90LIBS=""
fi

CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_MOD_EXT="mod"
