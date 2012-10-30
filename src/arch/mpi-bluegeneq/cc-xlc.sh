XLC_TYPICAL_PRE=/opt/ibmcmp/
if test -d /soft/compilers/ibmcmp-may2012
then
XLC_TYPICAL_PRE=/soft/compilers/ibmcmp-may2012
fi

XLC_PRE=$XLC_TYPICAL_PRE/vacpp/bg/12.1

XLF_TYPICAL_PRE=/soft/compilers/ibmcmp-feb2012/xlf/bg/14.1

XLC_TYPICAL_POST=bin/bg
XLC_POST=$XLC_TYPICAL_POST

# if no floor set, use typical floor path
if test -n "$BGQ_XLC_PRE"
then
  XLC_PRE=$BGQ_XLC_PRE
fi

XLC_F=$XLF_TYPICAL_PRE
CMK_CC="$XLC_PRE/${XLC_POST}xlc_r -qcpluscmt -qhalt=e $BGQ_INC -qnokeyword=__int128"
CMK_CXX="$XLC_PRE/${XLC_POST}xlC_r -qhalt=e $BGQ_INC -qnokeyword=__int128"
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_CF77="$XLC_F/${XLC_POST}xlf_r "
CMK_CF90="$XLC_F/${XLC_POST}xlf90_r  -qsuffix=f=f90" 
CMK_CF90_FIXED="$XLC_F/${XLC_POST}xlf90_r " 
CMK_C_OPTIMIZE='-O3 -Q'
CMK_CXX_OPTIMIZE='-O3 -Q'
CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
#CMK_NATIVE_CC="/opt/ibmcmp/vacpp/bg/9.0/bin/xlc"
#CMK_NATIVE_CXX="/opt/ibmcmp/vacpp/bg/9.0/bin/xlC"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"
CMK_RANLIB="ranlib"
CMK_F90LIBS="-L$XLC_F/lib -lxlf90 -lxlopt -lxl -lxlfmath"
