XLC_PRE=/soft/apps/ibmcmp
XLC_POST=bin/bg
XLC_F=$XLC_PRE/xlf/bg/11.1
CMK_CC="$XLC_PRE/vac/bg/9.0/${XLC_POST}xlc -qcpluscmt -qhalt=e -I$CHARMINC $BGP_INC"
CMK_CXX="$XLC_PRE/vacpp/bg/9.0/${XLC_POST}xlC -qhalt=e -I$CHARMINC $BGP_INC"
CMK_LD="$CMK_CC $BGP_LIB "
CMK_LDXX="$CMK_CXX $BGP_LIB"
CMK_CF77="$XLC_F/${XLC_POST}xlf "
CMK_CF90="$XLC_F/${XLC_POST}xlf90  -qsuffix=f=f90" 
CMK_CF90_FIXED="$XLC_PRE/xlf/bg/11.1/${XLC_POST}xlf90 " 
CMK_C_OPTIMIZE='-O3 -qstrict -Q  '
CMK_CXX_OPTIMIZE='-O3 -qstrict -Q '
CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
CMK_NATIVE_CC="gcc"
CMK_NATIVE_CXX="g++"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"
CMK_RANLIB="ranlib"
CMK_F90LIBS="-lxlf90 -lxlopt -lxl -lxlfmath"
