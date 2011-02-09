COMMENT="Enable 32-bit mode (-q32)"
if [ "$OBJECT_MODE" != "32" ]
then
	echo "WARNING: Charm built in 32-bit mode, but OBJECT_MODE=$OBJECT_MODE"
fi
CMK_CC='mpcc_r -q32 -qcpluscmt -qhalt=e '
CMK_CXX='mpCC_r -q32 -qhalt=e '
CMK_LD="mpcc_r -q32 -brtl"
CMK_LDXX="mpCC_r -q32 -brtl"
CMK_NATIVE_CC='xlc_r -q32'
CMK_NATIVE_LD='xlc_r -q32'
CMK_NATIVE_CXX='xlC_r -q32'
CMK_NATIVE_LDXX='xlC_r -q32'
CMK_SEQ_CC='xlc_r -q32'
CMK_SEQ_LD='xlc_r -q32'
CMK_SEQ_CXX='xlC_r -q32'
CMK_SEQ_LDXX='xlC_r -q32'
CMK_CF77='mpxlf_r -q32 '
CMK_CF90='mpxlf90_r -q32 -qsuffix=f=f90' 
CMK_CF90_FIXED='mpxlf90_r -q32 ' 
CMK_AR='ar -X 32 cq'
CMK_QT='aix'
CMK_NM='nm -X 32'
