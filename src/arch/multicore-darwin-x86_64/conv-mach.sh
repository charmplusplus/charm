CMK_MACOSX=1
CMK_DEFS=' -D_REENTRANT '

# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="/usr/bin/cpp -P "
CMK_CPP_C="clang -arch x86_64 -m64 -fPIC -E -mmacosx-version-min=10.7 "
CMK_CC="clang -arch x86_64 -m64 -dynamic -fno-common -mmacosx-version-min=10.7 $CMK_DEFS "
CMK_LD="clang -mmacosx-version-min=10.7 -Wl,-no_pie "
CMK_CXX="clang++ -arch x86_64 -m64 -fPIC -dynamic -fno-common -mmacosx-version-min=10.7 $CMK_DEFS -stdlib=libc++ "
CMK_CXXPP="clang++ -arch x86_64 -m64 -x clang++ -E -mmacosx-version-min=10.7 -stdlib=libc++ "
CMK_LDXX="$CMK_CXX -multiply_defined suppress -mmacosx-version-min=10.7 -Wl,-no_pie $CMK_DEFS -stdlib=libc++ "
CMK_XIOPTS=""
CMK_QT="generic64-light"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"

# Assumes GNU fortran compiler:
CMK_CF77="g77 -mmacosx-version-min=10.7"
CMK_CF90="g90 -mmacosx-version-min=10.7"

# setting for shared lib
# need -lc++ for c++ reference, and it needs to be put at very last
# of command line.
# Mac environment variable
test -z "$MACOSX_DEPLOYMENT_TARGET" && export MACOSX_DEPLOYMENT_TARGET=10.5
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_LIBS="-lc++"
CMK_LD_SHARED_ABSOLUTE_PATH=true
