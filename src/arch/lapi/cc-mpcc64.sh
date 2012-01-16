COMMENT="Enable 64-bit mode (-q64)"

if test $isAIX = true
then
CMK_MPCC="mpcc_r"
CMK_MPCXX="mpCC_r"
CMK_LD_OPT="-brtl"
CMK_ARNM_OPT="-X 64"
else
# assume Linux
CMK_MPCC="mpcc"
CMK_MPCXX="mpCC"
CMK_LD_OPT=""
CMK_ARNM_OPT=""
fi

CMK_CC="$CMK_MPCC -q64 -qcpluscmt -qhalt=e "
CMK_CXX="$CMK_MPCXX -q64 -qhalt=e "
CMK_LD="$CMK_MPCC -q64 $CMK_LD_OPT"
CMK_LDXX="$CMK_MPCXX -q64 $CMK_LD_OPT"

CMK_SEQ_CC='xlc_r -q64'
CMK_SEQ_LD='xlc_r -q64'
CMK_SEQ_CXX='xlC_r -q64'
CMK_SEQ_LDXX='xlC_r -q64'
CMK_CF77='mpxlf_r -q64 '
CMK_CF90='mpxlf90_r -q64 -qsuffix=f=f90' 
CMK_CF90_FIXED='mpxlf90_r -q64 ' 
CMK_C_OPTIMIZE='-O3 -qstrict -qnohot '
CMK_CXX_OPTIMIZE='-O3 -qstrict -qnohot '

CMK_AR="ar $CMK_ARNM_OPT cq"
CMK_QT='aix64-light'
CMK_NM="nm $CMK_ARNM_OPT"
