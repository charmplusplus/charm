BGL_TYPICAL_FLOOR=/bgl/BlueLight/ppcfloor

#BGL_SRC=/bgd-public/chuang10/bglsw
#BGL_INSTALL=/bgd-public/chuang10/bglsw-install
#BGL_FLOOR=/auto/BGX/floor

# if no floor set, use typical floor path
if test -z "$BGL_FLOOR"
then
  BGL_FLOOR=$BGL_TYPICAL_FLOOR
fi

# if no install path (for experimental) set, use floor
if test -z "$BGL_INSTALL"
then
  BGL_INSTALL=$BGL_FLOOR
fi

BGL_BIN=$BGL_FLOOR/blrts-gnu/bin
BGL_INC="-I$BGL_INSTALL/bglsys/include -I/gsa/yktgsa/home/g/m/gmartyna/dev/myincs"
#-I$BGL_INSTALL/blrts-gnu/powerpc-bgl-blrts-gnu/sys-include/" 

#BGL_LIB="-L$BGL_INSTALL/bglsys/lib -L/gsa/watgsa/home/s/a/sameer77/bgx/new2/bglsw-install/blrts-gnu/powerpc-bgl-blrts-gnu/lib -lc"

BGL_LIB="-L$BGL_INSTALL/bglsys/lib -L/gsa/yktgsa/home/g/m/gmartyna/dev/mylibs"        #" -L$BGL_INSTALL/blrts-gnu/powerpc-bgl-blrts-gnu/lib"

# test if compiler binary present
if test ! -x $BGL_BIN/powerpc-bgl-blrts-gnu-g++
then
 echo "ERROR: Invalid BGL_INSTALL or BGL_FLOOR, C/C++ compiler missing"
 exit 1
fi

OPTS_CPP="$OPTS_CPP"
GCC_OPTS="-gdwarf-2 $BGL_INC -I$BGL_INSTALL/blrts-gnu/powerpc-bgl-blrts-gnu/sys-include"
GXX_OPTS="$GCC_OPTS -Wno-deprecated"
OPTS_LD="$OPTS_LD"

CMK_CPP_CHARM="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -P"
CMK_CPP_C="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -E "
CMK_CXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ $GXX_OPTS "
CMK_GCXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ $GXX_OPTS "
CMK_CC="$BGL_BIN/powerpc-bgl-blrts-gnu-gcc $GCC_OPTS "
CMK_CXXPP="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ -E "
CMK_CF77="$BGL_BIN/powerpc-bgl-blrts-gnu-g77 "
CMK_CF90='f90'
CMK_RANLIB="$BGL_BIN/powerpc-bgl-blrts-gnu-ranlib "
CMK_AR="$BGL_BIN/powerpc-bgl-blrts-gnu-ar q "
CMK_LD="$CMK_CC $BGL_LIB"
CMK_LDXX="$CMK_CXX $BGL_LIB"
CMK_LIBS='-lckqt -lcommlib.rts -lrts.rts -ldevices.rts'
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
