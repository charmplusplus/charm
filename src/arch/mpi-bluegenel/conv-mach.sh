BGL_TYPICAL_FLOOR=/bgl/BlueLight/ppcfloor

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
BGL_INC=$BGL_INSTALL/bglsys/include
BGL_LIB=$BGL_INSTALL/bglsys/lib

# test if compiler binary present
if test ! -x $BGL_BIN/powerpc-bgl-blrts-gnu-g++
then
 echo "ERROR: Invalid BGL_INSTALL or BGL_FLOOR, C/C++ compiler missing"
 exit 1
fi

OPTS_CPP="$OPTS_CPP -I$BGL_INC "
GCC_OPTS="-gdwarf-2 -Wno-deprecated"
OPTS_LD="$OPTS_LD -L$BGL_LIB "

CMK_CPP_CHARM="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -P"
CMK_CPP_C="$BGL_BIN/powerpc-bgl-blrts-gnu-cpp -E "
CMK_CC="$BGL_BIN/powerpc-bgl-blrts-gnu-gcc $GCC_OPTS "
CMK_CXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ $GCC_OPTS "
CMK_CXXPP="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ -E "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "
CMK_LIBS='-lckqt -lmpich.rts -lmsglayer.rts -lrts.rts -ldevices.rts -lrts.rts'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_RANLIB="$BGL_BIN/powerpc-bgl-blrts-gnu-ranlib "
CMK_AR="$BGL_BIN/powerpc-bgl-blrts-gnu-ar q "
CMK_NM="$BGL_BIN/powerpc-bgl-blrts-gnu-nm"

#CMK_SEQ_LIBS=''
#CMK_SEQ_CC="$BGL_BIN/powerpc-bgl-blrts-gnu-gcc -Wno-deprecated "
#CMK_SEQ_LD="$CMK_SEQ_CC"
#CMK_SEQ_CXX="$BGL_BIN/powerpc-bgl-blrts-gnu-g++ -Wno-deprecated "
#CMK_SEQ_LDXX="$CMK_SEQ_CXX"

# native compiler
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'

CMK_QT="generic"

# fortran
#CMK_CF77="$BGL_BIN/powerpc-bgl-blrts-gnu-g77 "
#CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
XLC_F=/opt/ibmcmp/xlf/9.1
XLC_POST=bin/blrts_
CMK_CF77="$XLC_F/${XLC_POST}xlf "
CMK_CF90="$XLC_F/${XLC_POST}xlf90  -qsuffix=f=f90" 
CMK_CF90_FIXED="$XLC_F/${XLC_POST}xlf90 " 
CMK_F90LIBS="-L$XLC_F/blrts_lib -lxlf90 -lxlopt -lxl -lxlfmath"
CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
