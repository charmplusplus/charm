CMK_CPP_CHARM='cpp -P'
CMK_CPP_C='gcc'
CMK_CC='gcc '
CMK_CXX='g++ '
CMK_LD="eval $CMK_CC "

CMK_CPP_C_FLAGS="-E"
CMK_CC_FLAGS="-D_REENTRANT -I/usr/opt/rms/include"
CMK_CXX_FLAGS="-D_REENTRANT -I/usr/opt/rms/include"
CMK_LD_FLAGS="$CMK_CC_FLAGS"

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-rpath $CHARMLIBSO/"
CMK_QT='gcc'
CMK_XIOPTS='-ansi'

# fortran compilers
CMK_CF77='f77 -automatic'
CMK_CF90='f90 -automatic'
CMK_F90LIBS="-lUfor -lfor -lFutil"
CMK_F77LIBS="$CMK_F90LIBS"

CMK_COMPILER='gcc'
