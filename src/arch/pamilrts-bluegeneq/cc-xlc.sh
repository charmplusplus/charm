CMK_CC="bgxlc_r -qcpluscmt -qhalt=e $BGQ_INC -qnokeyword=__int128"
CMK_CXX="bgxlC_r -qhalt=e $BGQ_INC -qnokeyword=__int128"
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_CF77="bgxlf_r "
CMK_CF90="bgxlf90_r  -qsuffix=f=f90" 
CMK_CF90_FIXED="bgxlf90_r " 
CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'
CMK_ENABLE_C11='-qlanglvl=extc1x'
CMK_ENABLE_CPP11='-qlanglvl=extended0x'

CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"
CMK_RANLIB="ranlib"
CMK_F90LIBS="-lxlf90 -lxlopt -lxl -lxlfmath"

