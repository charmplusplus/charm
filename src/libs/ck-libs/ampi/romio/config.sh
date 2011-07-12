#!/bin/sh 
CHARMDIR=`util/charmdir`
./configure -mpiincdir=$CHARMDIR/include -mpilib=$CHARMDIR/lib/libmoduleampi.a -mpibindir=$CHARMDIR/bin -cc=$CHARMDIR/bin/ampicc -fc=$CHARMDIR/bin/ampif77 -f90=$CHARMDIR/bin/ampif90
