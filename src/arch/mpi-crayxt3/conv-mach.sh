CMK_DEFS="-I/opt/rs/1.8/mpich2-64/T/include "
CMK_LD_DEFS="-L/opt/rs/1.8/mpich2-64/T/lib "

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="mpicc -E $CMK_DEFS "
CMK_CXXPP="mpicxx -E $CMK_DEFS "
CMK_CC="mpicc -I/usr/include $CMK_DEFS "
CMK_CXX="mpicxx  -I/usr/include $CMK_DEFS "
CMK_LD="$CMK_CC $CMK_LD_DEFS"
CMK_LDXX="$CMK_CXX $CMK_LD_DEFS"
CMK_LIBS="-lpthread "

CMK_CF77="pgf77 "
CMK_CF90="pgf90 "

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_SEQ_CC="pgcc "
CMK_SEQ_LD="pgcc "
CMK_SEQ_CXX="pgCC "
CMK_SEQ_LDXX="pgCC "
CMK_SEQ_LIBS=""

CMK_RANLIB="ranlib"
CMK_QT="generic64"

CMK_F90LIBS=""
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"

