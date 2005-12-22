# GNU f95

CMK_CC="$CMK_CC -DCMK_G95"
CMK_CXX="$CMK_CXX -DCMK_G95"
CMK_CF90=`which g95 2>/dev/null`
CMK_FPP="/lib/cpp -P -CC"
CMK_CF90="$CMK_CF90 -fpic"
CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "
# find f90 librarya:  
#it can be at g95-install/lib/gcc-lib/i686-pc-linux-gnu/4.0.1
F90DIR=`which g95 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
F90LIBDIR="`dirname $F90DIR`/../lib/*/*/*"
F90LIBDIR=`cd $F90LIBDIR && pwd`
CMK_F90LIBS="-L$F90LIBDIR -lf95 "
CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"

