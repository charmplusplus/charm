# g95.org

CMK_CC="$CMK_CC -DCMK_G95"
CMK_CXX="$CMK_CXX -DCMK_G95"

if test -n "$CMK_MACOSX64" 
then
CMK_F90FLAGS="$CMK_F90FLAGS -m64"
fi

if test -n "$CMK_MACOSX"
then
CMK_F90FLAGS="$CMK_F90FLAGS -fno-common"
fi

CMK_CF90=`which g95 2>/dev/null`
CMK_FPP="/lib/cpp -P -CC"
CMK_CF90="$CMK_CF90 $CMK_F90FLAGS -fPIC -fno-second-underscore -fdollar-ok"
CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "
# find f90 library:
#it can be at g95-install/lib/gcc-lib/i686-pc-linux-gnu/4.0.1
F90DIR=`which g95 2> /dev/null`
readlinkcmd=`which readlink 2> /dev/null`
if test -h "$F90DIR" && test -x "$readlinkcmd"
then
  LINKDIR=`readlink $F90DIR`
  case "$LINKDIR" in
  \/*)
       F90DIR=$LINKDIR
       ;;
  *)
       basedir=`dirname $F90DIR`
       F90DIR="$basedir/$LINKDIR"
       ;;
  esac
fi
F90DIR="`dirname $F90DIR`"
F90LIBDIR=`cd $F90DIR/../lib/gcc-lib/*/*; pwd`
CMK_F90LIBS="-L$F90LIBDIR -lf95 -lgcc_eh"

CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
