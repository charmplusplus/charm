CMK_MACOSX=1

# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="clang -m64 -fPIC -E -mmacosx-version-min=10.7 "
CMK_CC="clang -m64 -dynamic -fno-common -mmacosx-version-min=10.7 "
CMK_LD="clang -mmacosx-version-min=10.7 -Wl,-no_pie "
CMK_CXX="clang++ -m64 -fPIC -dynamic -fno-common -mmacosx-version-min=10.7 -stdlib=libc++ "
CMK_CXXPP="clang++ -m64 -x clang++ -E -mmacosx-version-min=10.7 -stdlib=libc++ "
CMK_LDXX="clang++ -multiply_defined suppress -mmacosx-version-min=10.7 -Wl,-no_pie -stdlib=libc++ "
CMK_XIOPTS=""
CMK_QT="generic64-light"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"

# Assumes GNU fortran compiler:
CMK_CF77="g95 -arch x86_64 -mmacosx-version-min=10.7"
CMK_CF90="g95 -arch x86_64 -mmacosx-version-min=10.7"

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last 
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.5
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
