CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C='mpic++11'
CMK_CXX='mpic++11'
CMK_CC='mpicc'
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_CPP_C_FLAGS='-E'

CMK_CC_FLAGS='-Wno-deprecated-declarations'
CMK_CXX_FLAGS='-Wno-deprecated-declarations'
CMK_LD_FLAGS=''
CMK_LDXX_FLAGS=''

CMK_C_OPTIMIZE='-O3 -ffast-math '
CMK_CXX_OPTIMIZE='-O3 -ffast-math '
CMK_QT="generic64-light"

CMK_CF77=""
CMK_CF90=""
CMK_CF90_FIXED=""

CMK_COMPILER='bgclang'
