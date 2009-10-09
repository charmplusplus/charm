

CMK_CC="$CMK_CC -mno-cygwin -I/usr/local/mingw/include "
CMK_CXX="$CMK_CXX -mno-cygwin -I/usr/local/mingw/include "
CMK_LD="$CMK_LD -mno-cygwin -L/usr/local/mingw/lib "
CMK_LDXX="$CMK_LDXX -mno-cygwin -L/usr/local/mingw/lib "
CMK_SYSLIBS="-lwsock32 -lpsapi"

CMK_LIBS=""
CMK_QT="none"

CMK_NATIVE_CC="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_LDXX="g++ "
CMK_SEQ_LIBS="$CMK_SEQ_LIBS -lwsock32"
