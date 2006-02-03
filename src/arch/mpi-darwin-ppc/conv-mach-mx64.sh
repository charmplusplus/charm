#default gm dir
#guess where the gm.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /turing/software/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/turing/software/mx/include"
    CMK_LIBDIR="-L/turing/software/mx/lib"
  fi
fi
CMK_SYSLIBS="$CMK_SYSLIBS -lmyriexpress"

OPTS_CC="-m64 $OPTS_CC"
OPTS_CXX="-m64 $OPTS_CXX"

CMK_QT="generic64-light"

# native compilers
CMK_NATIVE_CC="gcc -m64"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_CXX="g++ -m64"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"

