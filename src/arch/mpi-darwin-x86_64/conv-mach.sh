. $CHARMINC/cc-mpiopts.sh

CMK_MACOSX=1

CMK_DEFS="$CMK_DEFS -mmacosx-version-min=10.7 -D_DARWIN_C_SOURCE"

CMK_AMD64="-dynamic -fPIC -fno-common -Wno-deprecated-declarations"

CMK_CC="$MPICC "
CMK_CXX="$MPICXX "

CMK_CPP_C_FLAGS="$CMK_CPP_C_FLAGS"
CMK_CC_FLAGS="$CMK_CC_FLAGS $CMK_AMD64"

CMK_CLANG_CXX_FLAGS="-stdlib=libc++"
CMK_REAL_COMPILER=`$MPICXX -show 2>/dev/null | cut -d' ' -f1 `
case "$CMK_REAL_COMPILER" in
  g++)
    CMK_CXX_FLAGS="$CMK_CXX_FLAGS $CMK_AMD64"
    CMK_COMPILER='gcc'
    ;;
  clang)
    CMK_CXX_FLAGS="$CMK_CXX_FLAGS $CMK_AMD64 $CMK_CLANG_CXX_FLAGS"
    CMK_COMPILER='clang'
    ;;
esac

CMK_XIOPTS=""

CMK_NATIVE_CC='clang'
CMK_NATIVE_LD='clang'
CMK_NATIVE_CXX='clang++'
CMK_NATIVE_LDXX='clang++'
CMK_NATIVE_LIBS=""

CMK_NATIVE_CC_FLAGS="$CMK_GCC64"
CMK_NATIVE_LD_FLAGS="-Wl,-no_pie $CMK_GCC64"
CMK_NATIVE_CXX_FLAGS="$CMK_GCC64 -stdlib=libc++"
CMK_NATIVE_LDXX_FLAGS="-Wl,-no_pie $CMK_GCC64 -stdlib=libc++"

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
    . $CHARMINC/conv-mach-gfortran.sh
else
    CMK_CF77="g77 "
    CMK_CF90="f90 "
    CMK_CF90_FIXED="$CMK_CF90 -W132 "
    CMK_F90LIBS="-lf90math -lfio -lU77 -lf77math "
    CMK_F77LIBS="-lg2c "
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC="-p"
fi

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.7
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
