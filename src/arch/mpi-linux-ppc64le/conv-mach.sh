. $CHARMINC/cc-mpiopts.sh

CMK_QT='generic64-light'


CMK_CC="$MPICC"
CMK_CXX="$MPICXX"

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_NATIVE_CC='gcc'
CMK_NATIVE_CXX='g++'
CMK_NATIVE_LD='gcc'
CMK_NATIVE_LDXX='g++'
CMK_NATIVE_LIBS=''

CMK_NATIVE_FLAGS='-fPIC'

CMK_NATIVE_CC_FLAGS="$CMK_NATIVE_FLAGS"
CMK_NATIVE_LD_FLAGS="$CMK_NATIVE_FLAGS"
CMK_NATIVE_CXX_FLAGS="$CMK_NATIVE_FLAGS"
CMK_NATIVE_LDXX_FLAGS="$CMK_NATIVE_FLAGS"

# Fortran
CMK_CXX_IS_GCC=`$MPICXX -V 2>&1 | grep 'g++' `
CMK_CXX_IS_ICC=`$MPICXX -V 2>&1 | grep Intel `
CMK_CXX_IS_NVHPC=`$MPICXX -V 2>&1 | grep 'nvc++' `
if test -n "$CMK_CXX_IS_GCC"
then
    . $CHARMINC/conv-mach-gfortran.sh
elif test -n "$CMK_CXX_IS_ICC"
then
    . $CHARMINC/conv-mach-ifort.sh
elif test -n "$CMK_CXX_IS_NVHPC"
then
    . $CHARMINC/conv-mach-nvfortran.sh
fi

CMK_COMPILER='mpicc'
