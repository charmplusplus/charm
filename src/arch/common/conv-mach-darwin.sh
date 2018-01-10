CMK_MACOSX=1

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_QT='generic64-light'
CMK_XIOPTS=''

CMK_CC_FLAGS="$CMK_CC_FLAGS -fPIC -dynamic -fno-common "
CMK_LD_FLAGS="$CMK_LD_FLAGS -Wl,-no_pie "
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -fPIC -dynamic -fno-common -stdlib=libc++ "
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -multiply_defined suppress -Wl,-no_pie -stdlib=libc++ "

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.7
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS='-lc++'
CMK_LD_SHARED_ABSOLUTE_PATH=true

CMK_DEFS='-mmacosx-version-min=10.7 -D_DARWIN_C_SOURCE'

# Assumes gfortran compiler:
CMK_CF77="gfortran -mmacosx-version-min=10.7"
CMK_CF90="gfortran -mmacosx-version-min=10.7"
