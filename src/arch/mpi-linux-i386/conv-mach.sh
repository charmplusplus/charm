CMK_DEFS="$CMK_DEFS -m32"
CMK_FDEFS="$CMK_FDEFS -m32"

. $CHARMINC/cc-mpi.sh

CMK_QT='i386-gcc'

CMK_NATIVE_CC='gcc '
CMK_NATIVE_CXX='g++ '
CMK_NATIVE_LD='gcc'
CMK_NATIVE_LDXX='g++'
CMK_NATIVE_LIBS=''

CMK_CF77='f77'
CMK_CF90='f90'
CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
