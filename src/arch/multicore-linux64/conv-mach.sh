. $CHARMINC/cc-gcc.sh

CMK_DEFS=' -D_REENTRANT '

CMK_XIOPTS=''
CMK_LIBS="-lpthread $CMK_LIBS"

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
    . $CHARMINC/conv-mach-gfortran.sh
else
    CMK_CF77='g77 '
    CMK_CF90='f90 '
    CMK_CF90_FIXED="$CMK_CF90 -W132 "
    CMK_F90LIBS='-lf90math -lfio -lU77 -lf77math '
    CMK_F77LIBS='-lg2c '
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC='-p'
fi

CMK_QT='generic64-light'

CMK_SMP='1'
CMK_NO_PARTITIONS="1"
