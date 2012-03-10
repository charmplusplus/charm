# gfortran.org

CMK_CC="$CMK_CC -DCMK_GFORTRAN"
CMK_CXX="$CMK_CXX -DCMK_GFORTRAN"

if test -n "$CMK_MACOSX64"
then
CMK_F90FLAGS="$CMK_F90FLAGS -m64"
CMK_F77FLAGS="$CMK_F90FLAGS -m64"
fi

if test -n "$CMK_MACOSX"
then
CMK_F90FLAGS="$CMK_F90FLAGS -fno-common"
CMK_F77FLAGS="$CMK_F90FLAGS -fno-common"
fi

CMK_FPP="/lib/cpp -P -CC"

CMK_CF90=`which gfortran 2>/dev/null`
CMK_CF90="$CMK_CF90 $CMK_F90FLAGS -fPIC -fno-second-underscore -fdollar-ok" 
CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "
# find f90 library:
#it can be at gfortran-install/lib/gcc-lib/i686-pc-linux-gnu/4.0.1
F90DIR=`which gfortran 2> /dev/null`
#F90DIR=$HOME/gfortran-install/bin/gfortran
readlinkcmd=`which readlink 2> /dev/null`
if test -h "$F90DIR" && test -x "$readlinkcmd"
then
  F90DIR=`readlink $F90DIR`
  test `basename $F90DIR` = "$F90DIR" && F90DIR=`which gfortran 2> /dev/null`
fi
F90DIR="`dirname $F90DIR`"

# test some well-known place
if test -f /usr/lib/libgfortran.a
then
  F90LIBDIR=/usr/lib
else
  f95target=`gfortran -v 2>&1 | grep Target | cut -f2 -d' '`
  f95version=`gfortran -v 2>&1 | grep 'gcc version' | cut -d' ' -f3`
  F90LIBDIR=`cd $F90DIR/../lib/gcc/$f95target/$f95version/ 2>/dev/null && pwd`
  test -z "$F90LIBDIR" && F90LIBDIR=`cd $F90DIR/../lib/$f95target/gcc/$f95version/ 2>/dev/null && pwd`
  #F90LIBDIR=`cd $F90DIR/../lib/gcc/ia64-unknown-linux-gnu/4.1.0; pwd`
fi
test -n "$F90LIBDIR" && CMK_F90LIBS="-L$F90LIBDIR -lgfortran -lgcc_eh"

CMK_CF77=$CMK_CF90
CMK_F77LIBS=$CMK_F90LIBS

CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"

