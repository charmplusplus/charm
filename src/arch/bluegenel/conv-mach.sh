BGL_SRC=/bgd-public/chuang10/bglsw
BGL_INSTALL=/bgd-public/chuang10/bglsw-install
BGL_FLOOR=/auto/BGX/floor
[ -z "$BGL_INSTALL" ] && echo "ERROR: BL_INSTALL not defined!" && exit 1

BGL_BASE=$BGL_INSTALL
BGL_BIN=$BGL_FLOOR/blrts-gnu/bin
BGL_INC="-I$BGL_SRC/commlib/BGLML -I$BGL_INSTALL/bglsys/include -I$BGL_INSTALL/bglsys/include/bglml"
BGL_LIB="-L$BGL_INSTALL/bglsys/lib"

OPTS_CPP="$OPTS_CPP $BGL_INC "
GCC_OPTS="-gdwarf-2 -Wno-deprecated"
OPTS_LD="$OPTS_LD $BGL_LIB "

CMK_CPP_CHARM="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -P"
CMK_CPP_C="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -E "
CMK_CXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ $GCC_OPTS "
CMK_CC="$BGL_BIN/powerpc-bgl-blrts-gnu-gcc $GCC_OPTS "
CMK_CXXPP="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ -E "
CMK_CF77="$BGL_BIN/powerpc-bgl-blrts-gnu-g77 "
CMK_CF90='f90'
CMK_RANLIB="$BGL_BIN/powerpc-bgl-blrts-gnu-ranlib "
CMK_AR="$BGL_BIN/powerpc-bgl-blrts-gnu-ar q "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "
CMK_LIBS='-lckqt -lbglml.rts -lrts.rts -ldevices.rts'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
#CMK_SEQ_LIBS=''
#CMK_SEQ_CC="$BGL_BIN/powerpc-bgl-blrts-gnu-gcc -Wno-deprecated "
#CMK_SEQ_LD="$CMK_SEQ_CC"
#CMK_SEQ_CXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ -Wno-deprecated "
#CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"
CMK_QT="generic"
