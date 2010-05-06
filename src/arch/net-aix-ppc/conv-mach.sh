isAIX=true

CMK_CPP_CHARM='/usr/lib/cpp'
CMK_CPP_C='xlc -E'
CMK_CC='xlc_r -qcpluscmt '
CMK_CXX='xlC_r -qstaticinline '
CMK_LD="xlc_r -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000 "
CMK_LDXX="xlC_r -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000"
CMK_CXXPP='xlC -E'
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lhC'
CMK_LD_SHARED='-G'
CMK_QT='aix-light'
CMK_XIOPTS=''

CMK_NATIVE_CXX='xlC -D_H_UNISTD -DYY_NEVER_INTERACTIVE=1 -qstaticinline '
CMK_NATIVE_LDXX='xlC'

CMK_CF77='xlf77_r'
CMK_CF90='xlf90_r -qsuffix=f=f90'
CMK_F90LIBS='-lxlf90_r -lhC'
