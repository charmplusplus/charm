XLC_PRE=/soft/apps/ibmcmp
XLC_POST=bin/bg
XLC_F=$XLC_PRE/xlf/bg/11.1/
CMK_CC="$XLC_PRE/vacpp/bg/9.0/${XLC_POST}xlc -qcpluscmt -qhalt=e $BGP_INC"
CMK_CXX="$XLC_PRE/vacpp/bg/9.0/${XLC_POST}xlC -qhalt=e $BGP_INC"
CMK_LD="$CMK_CC $BGP_LIB "
CMK_LDXX="$CMK_CXX  $BGP_LIB"
CMK_CF77="$XLC_F/${XLC_POST}xlf "
CMK_CF90="$XLC_F/${XLC_POST}xlf90  -qsuffix=f=f90" 
CMK_CF90_FIXED="$XLC_PRE/xlf/8.1/${XLC_POST}xlf90 " 
CMK_C_OPTIMIZE='-O2 -qstrict -Q  '
CMK_CXX_OPTIMIZE='-O2 -qstrict -Q '
CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
#CMK_NATIVE_CC="/soft/apps/ibmcmp/vacpp/bg/9.0/bin/xlc"
#CMK_NATIVE_CXX="/soft/apps/ibmcmp/vacpp/bg/9.0/bin/xlC"
CMK_NATIVE_LD="/soft/apps/ibmcmp/vacpp/bg/9.0/bin/xlc"      #"$CMK_NATIVE_CC"
CMK_NATIVE_LDXX="/soft/apps/ibmcmp/vacpp/bg/9.0/bin/xlC"    #"$CMK_NATIVE_CXX"
CMK_RANLIB="ranlib"
CMK_F90LIBS="-L$XLC_F/lib -lxlf90 -lxlopt -lxl -lxlfmath"
