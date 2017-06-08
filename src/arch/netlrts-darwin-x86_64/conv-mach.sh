. $CHARMINC/cc-clang.sh

CMK_MACOSX=1

# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="clang -fPIC -E -mmacosx-version-min=10.7 "
CMK_CC="clang -dynamic -fno-common -mmacosx-version-min=10.7 -Wno-deprecated-declarations "
CMK_LD="clang -mmacosx-version-min=10.7 -Wl,-no_pie "
CMK_CXX="clang++ -fPIC -dynamic -fno-common -mmacosx-version-min=10.7 -stdlib=libc++ -Wno-deprecated-declarations "
CMK_LDXX="clang++ -multiply_defined suppress -mmacosx-version-min=10.7 -Wl,-no_pie -stdlib=libc++ "
CMK_XIOPTS=""
CMK_QT="generic64-light"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"

CMK_WARNINGS_ARE_ERRORS="-Werror"

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

CMK_USING_CLANG="1"
