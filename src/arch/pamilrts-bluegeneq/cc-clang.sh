CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="bgclang++ -E "
CMK_CXX="bgclang++ -Wno-deprecated-declarations "
CMK_CC="bgclang -Wno-deprecated-declarations "
CMK_CXXPP="bgclang++ -E "
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_C_OPTIMIZE='-O3 -ffast-math '
CMK_CXX_OPTIMIZE='-O3 -ffast-math '
OPTS_LD="$OPTS_LD "
OPTS_CC="$OPTS_CC -x c "
CMK_QT="generic64-light"

CMK_USING_BGCLANG="1"

