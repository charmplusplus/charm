#!/bin/sh 
CHARMDIR=`util/charmdir`
ARCH=`util/tarch`
./configure -mpiincdir=$CHARMDIR/include -mpilib=$CHARMDIR/lib/libmoduleampi.a -mpibindir=$CHARMDIR/bin -cc=$CHARMDIR/bin/mpicc -fc=$CHARMDIR/bin/mpif77 -f90=$CHARMDIR/bin/mpif90 -cflags="" -fflags=""
if [ $? = 0 ]; then 
  cp include/mpio.h include/mpiof.h $CHARMDIR/include
  make
  cp lib/$ARCH/libmpio.a $CHARMDIR/lib/libampiromio.a
fi
echo "configured"
