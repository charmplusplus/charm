CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="bgclang++ -E "
CMK_CXX="bgclang++  "
CMK_CC="bgclang "
CMK_CXXPP="bgclang++ -E "
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_C_OPTIMIZE='-O3 -ffast-math '
CMK_CXX_OPTIMIZE='-O3 -ffast-math '
OPTS_LD="$OPTS_LD "
CMK_AR="bgclang-ar q"
CMK_NM='bgclang-nm '
CMK_QT="generic64-light"

