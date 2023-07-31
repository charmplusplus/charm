. $CHARMINC/cc-mpiopts.sh

CMK_GCC64='-fPIC'

case "$CMK_REAL_COMPILER" in
g++) CMK_AMD64_CC="$CMK_GCC64"; CMK_AMD64_CXX="$CMK_GCC64" ;;
pgCC|pgc++|nvc++)  CMK_AMD64_CC='-fPIC'; CMK_AMD64_CXX='-fPIC -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std ' ;;
charmc)  echo "Error> charmc can not call AMPI's mpicxx/mpiCC wrapper! Please fix your PATH."; exit 1 ;;
esac

CMK_CC="$MPICC "
CMK_CXX="$MPICXX "

CMK_CC_FLAGS="$CMK_CC_FLAGS $CMK_AMD64_CC"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS $CMK_AMD64_CXX"

#CMK_SYSLIBS="-lmpich"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_NATIVE_CC='gcc'
CMK_NATIVE_LD='gcc'
CMK_NATIVE_CXX='g++'
CMK_NATIVE_LDXX='g++'
CMK_NATIVE_LIBS=''

CMK_NATIVE_CC_FLAGS="$CMK_GCC64 "
CMK_NATIVE_LD_FLAGS="$CMK_GCC64 "
CMK_NATIVE_CXX_FLAGS="$CMK_GCC64 "
CMK_NATIVE_LDXX_FLAGS="$CMK_GCC64 "

# fortran compiler
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
