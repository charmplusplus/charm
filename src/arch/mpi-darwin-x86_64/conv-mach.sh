. $CHARMINC/cc-mpiopts.sh

CMK_MACOSX=1

CMK_AMD64="-dynamic -fPIC -fno-common -mmacosx-version-min=10.7 -Wno-deprecated-declarations"

CMK_CC="$MPICC "
CMK_CXX="$MPICXX "

CMK_CPP_C_FLAGS="$CMK_CPP_C_FLAGS -mmacosx-version-min=10.7"
CMK_CC_FLAGS="$CMK_CC_FLAGS $CMK_AMD64"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS $CMK_AMD64"

CMK_XIOPTS=""

CMK_NATIVE_CC="clang $CMK_GCC64 "
CMK_NATIVE_LD="clang -Wl,-no_pie $CMK_GCC64 "
CMK_NATIVE_CXX="clang++ $CMK_GCC64 -stdlib=libc++ "
CMK_NATIVE_LDXX="clang++ -Wl,-no_pie $CMK_GCC64 -stdlib=libc++ "
CMK_NATIVE_LIBS=""

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
CMK_USING_CLANG="1"
