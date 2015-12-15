HPC_SDK="c:\Program Files\Microsoft MPI"
HPC_SDK=`cygpath -d "$HPC_SDK"`

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="gcc -E "
CMK_CXXPP="g++ -x c++ -E "
CMK_CC="gcc -mno-cygwin -I/usr/local/mingw/include -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_CXX="g++ -mno-cygwin -I/usr/local/mingw/include -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_LD="$CMK_CC -L/usr/local/mingw/lib "
CMK_LDXX="$CMK_CXX -L/usr/local/mingw/lib "
CMK_LIBS=""
CMK_SYSLIBS="`cygpath -u "$HPC_SDK\Lib\i386"`/msmpi.lib -lwsock32 -lpsapi"
CMK_QT="none"

CMK_NATIVE_CC="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_LDXX="g++ "
CMK_SEQ_LIBS="$CMK_SEQ_LIBS -lwsock32"

CMK_CF77="f77"
CMK_CF90="f90"
CMK_RANLIB="ranlib"
CMK_XIOPTS=""
CMK_F90LIBS="-lvast90 -lg2c"
CMK_MOD_EXT="vo"
CMK_POST_EXE=".exe"

