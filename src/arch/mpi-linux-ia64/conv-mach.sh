CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C='gcc -E'
CMK_CC="mpicc $CMK_INCDIR"
CMK_CXX="mpiCC $CMK_INCDIR"
CMK_CXXPP="$CMK_CXX -E "
CMK_XIOPTS=''
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt '

CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "

CMK_CF77="mpif77"
CMK_CF90="mpif90"
CMK_F90LIBS='-L/usr/absoft/lib -lf90math -lfio -lU77 -lf77math '
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"

