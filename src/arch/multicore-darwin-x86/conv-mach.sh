CMK_MACOSX=1
CMK_DEFS=' -D_REENTRANT '

# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="clang -m32 -fPIC -E -mmacosx-version-min=10.6 "
CMK_CC="clang -m32 -dynamic -fno-common -mmacosx-version-min=10.6 $CMK_DEFS "
CMK_LD="clang -m32 -mmacosx-version-min=10.6 "
CMK_CXX="clang++ -m32 -fPIC -dynamic -fno-common -mmacosx-version-min=10.6 $CMK_DEFS "
CMK_CXXPP="clang++ -m32 -x clang++ -E -mmacosx-version-min=10.6 "
CMK_LDXX="$CMK_CXX -m32 -multiply_defined suppress -mmacosx-version-min=10.6 $CMK_DEFS "
CMK_XIOPTS=""
CMK_QT="generic-light"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"

# Assumes GNU fortran compiler:
CMK_CF77="g77 -mmacosx-version-min=10.6"
CMK_CF90="g90 -mmacosx-version-min=10.6"

# setting for shared lib
# need -lstdc++ for c++ reference, and it needs to be put at very last 
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.3
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lstdc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
