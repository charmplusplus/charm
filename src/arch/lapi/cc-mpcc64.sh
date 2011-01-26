COMMENT="Enable 64-bit mode (-q64)"
CMK_CC='mpcc_r -q64 -qcpluscmt -qhalt=e '
CMK_CXX='mpCC_r -q64 -qstaticinline -qhalt=e '
CMK_LD="mpcc_r -q64 -brtl"
CMK_LDXX="mpCC_r -q64 -brtl"
CMK_SEQ_CC='xlc_r -q64'
CMK_SEQ_LD='xlc_r -q64'
CMK_SEQ_CXX='xlC_r -qstaticinline -q64'
CMK_SEQ_LDXX='xlC_r -q64'
CMK_CF77='mpxlf_r -q64 '
CMK_CF90='mpxlf90_r -q64 -qsuffix=f=f90' 
CMK_CF90_FIXED='mpxlf90_r -q64 ' 
CMK_C_OPTIMIZE='-O3 -qstrict -qnohot '
CMK_CXX_OPTIMIZE='-O3 -qstrict -qnohot '
CMK_AR='ar -X 64 cq'
CMK_QT='aix64-light'
CMK_NM='nm -X 64'
