CMK_CPP_CHARM='/usr/lib/cpp -P -D_NO_PROTO '
CMK_CPP_C='/usr/lib/cpp -P -D_NO_PROTO '
CMK_LDRO='ld -r -o '
CMK_LDRO_WORKS=0
CMK_CC='mpcc_r -qcpluscmt -qhalt=e '
CMK_CXX='mpCC_r -qhalt=e '
CMK_CXXPP='xlC -E '
CMK_LD="mpcc_r -brtl "
CMK_LDXX="mpCC_r -brtl "
CMK_C_OPTIMIZE='-O3 -qstrict -Q!  '
CMK_CXX_OPTIMIZE='-O3 -qstrict -Q! '
CMK_AR='ar cq'
CMK_RANLIB='true'
CMK_LIBS="-lckqt -lhC"
CMK_LD_SHARED='-G'
CMK_NM='/bin/nm'
CMK_NM_FILTER="grep ^_CK_ | cut -f 1 -d ' '"
if [ "$OBJECT_MODE" = "64" ]
then
	CMK_QT='aix64'
else
	CMK_QT='aix'
fi
CMK_XIOPTS=''

CMK_NATIVE_CC='xlc_r'
CMK_NATIVE_LD='xlc_r'
CMK_NATIVE_CXX='xlC_r '
CMK_NATIVE_LDXX='xlC_r'
CMK_NATIVE_LIBS=''

CMK_CF77='mpxlf_r'
CMK_CF90='mpxlf90_r -qsuffix=f=f90'
CMK_CF90_FIXED='mpxlf90_r '
CMK_F90LIBS='-lxlf90_r -lxlopt -lhC'
