#!/bin/sh 
CHARMDIR=`util/charmdir`
./configure -mpiincdir=$CHARMDIR/include -mpilib=$CHARMDIR/lib/libmoduleampi.a -mpibindir=$CHARMDIR/bin -cc=$CHARMDIR/bin/mpicc -fc=$CHARMDIR/bin/mpif77 -f90=$CHARMDIR/bin/mpif90
