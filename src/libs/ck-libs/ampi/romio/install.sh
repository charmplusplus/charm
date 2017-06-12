#!/bin/sh 
CHARMDIR="$( cd -P "$( dirname "$0" )/../../../../../" && pwd )"
ARCH=`util/tarch`
if test -f include/mpio.h ; then
  cp include/mpio.h include/mpiof.h $CHARMDIR/include && make && cp lib/$ARCH/libmpio.a $CHARMDIR/lib/libampiromio.a ;
fi
