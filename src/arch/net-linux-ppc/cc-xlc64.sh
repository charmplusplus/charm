COMMENT="Enable 64-bit mode (-q64)"
CMK_CC='xlc_r -q64 '
#CMK_CXX='xlC_r -q64 '
CMK_CXX='xlC_r -q64 '
CMK_C_OPTIMIZE='-O3 -qstrict '
CMK_CXX_OPTIMIZE='-O3 -qstrict '
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "

CMK_QT="generic64-light"

CMK_NATIVE_CC='xlc_r -q64'
CMK_NATIVE_LD='xlc_r -q64'
CMK_NATIVE_CXX='xlC_r -q64'
CMK_NATIVE_LDXX='xlC_r -q64'

CMK_CF77='xlf_r -q64 '
CMK_CF90='xlf90_r -q64 -qsuffix=f=f90' 
CMK_CF90_FIXED='xlf90_r -q64 ' 

CMK_AR='ar cq'
CMK_NM='nm '
