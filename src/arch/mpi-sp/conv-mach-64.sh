COMMENT="Enable 64-bit mode (-q64)"
CMK_CC='mpcc_r -q64 '
CMK_CXX='mpCC_r -q64 -qstaticinline '
CMK_LD="$CMK_CC -b64 "
CMK_LDXX="$CMK_CXX -b64 "
CMK_SEQ_CC='xlc_r -q64'
CMK_SEQ_LD='xlc_r -q64'
CMK_SEQ_CXX='xlC_r -qstaticinline -q64'
CMK_SEQ_LDXX='xlC_r -q64'
CMK_CF77='mpxlf_r -q64 '
CMK_CF90='mpxlf90_r -q64 -qsuffix=f=f90' 
CMK_CF90_FIXED='mpxlf90_r -q64 ' 
CMK_C_OPTIMIZE='-O3 -qstrict -Q  '
CMK_CXX_OPTIMIZE='-O3 -qstrict -Q '
CMK_AR='ar -X 64 cq'
