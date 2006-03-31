CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C='gcc -E'
CMK_CC='gcc -D_REENTRANT -I/usr/opt/rms/include '
CMK_CXX='g++ -D_REENTRANT -I/usr/opt/rms/include '
CMK_CXXPP="$CMK_CXX -x c++ -E  "
CMK_LD="eval $CMK_CC "
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
