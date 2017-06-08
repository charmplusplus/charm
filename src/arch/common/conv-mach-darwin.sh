CMK_MACOSX=1

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_QT='generic64-light'
CMK_XIOPTS=''

CMK_CPP_C="$CMK_CPP_C -fPIC "
CMK_CC="$CMK_CC -dynamic -fno-common "
CMK_LD="$CMK_LD -Wl,-no_pie "
CMK_CXX="$CMK_CXX -fPIC -dynamic -fno-common -stdlib=libc++ "
CMK_LDXX="$CMK_LDXX -multiply_defined suppress -Wl,-no_pie -stdlib=libc++ "

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.7
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS='-lc++'
CMK_LD_SHARED_ABSOLUTE_PATH=true

CMK_DEFS=' -mmacosx-version-min=10.7 '

# Assumes GNU fortran compiler:
CMK_CF77="g95 -mmacosx-version-min=10.7"
CMK_CF90="g95 -mmacosx-version-min=10.7"
