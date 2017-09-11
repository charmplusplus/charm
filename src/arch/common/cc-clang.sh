# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="clang"
CMK_CC="clang"
CMK_LD="clang"
CMK_CXX="clang++"
CMK_LDXX="clang++"

CMK_CPP_C_FLAGS="-E"

if [ -z "$CMK_MACOSX" ]; then
    CMK_CC_FLAGS="-Wno-deprecated-declarations"
    CMK_CXX_FLAGS="-Wno-deprecated-declarations"
fi

CMK_COMPILER='clang'
CMK_WARNINGS_ARE_ERRORS="-Werror"
