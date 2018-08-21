. $CHARMINC/cc-gcc.sh

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
    . $CHARMINC/conv-mach-gfortran.sh
else
    CMK_CF77='g77 '
    CMK_CF90='f90 '
    CMK_CF90_FIXED="$CMK_CF90 -W132 "
    CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
    CMK_F77LIBS='-lg2c '
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC='-p'
fi

if test -z "$USER_OPTS_LD"
then
    CMK_INCDIR="-I/usr/include/"
    CMK_LIBDIR="-L/usr/lib64/"
fi

CMK_LIBS="$CMK_LIBS -lucp  -luct -lucs -lucm -lpmix -lopen-pal -lopen-rte"

CMK_QT='generic64-light'
