XLC_PRE=/opt/ibmcmp
XLC_POST=bin/blrts_
CMK_CC="$XLC_PRE/vac/6.0/${XLC_POST}xlc -qcpluscmt "
CMK_CXX="$XLC_PRE/vacpp/6.0/${XLC_POST}xlC -qstaticinline "
CMK_LD="$CMK_CC  "
CMK_LDXX="$CMK_CXX  "
CMK_CF77="$XLC_PRE/xlf/8.1/${XLC_POST}xlf "
CMK_CF90="$XLC_PRE/xlf/8.1/${XLC_POST}xlf90  -qsuffix=f=f90" 
CMK_CF90_FIXED="$XLC_PRE/xlf/8.1/${XLC_POST}xlf90 " 
CMK_C_OPTIMIZE='-O3 -qstrict -Q  '
CMK_CXX_OPTIMIZE='-O3 -qstrict -Q '
CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
CMK_SEQ_CXX="$CMK_CXX"
CMK_SEQ_CC="$CMK_CC"
CMK_RANLIB="ranlib"
