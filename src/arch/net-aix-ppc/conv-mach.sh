isAIX=true

CMK_CPP_CHARM='/usr/lib/cpp -P -D_NO_PROTO '
CMK_CPP_C='xlc -q32 -E'
CMK_CC='xlc_r -q32 -qcpluscmt -qhalt=e '
CMK_CXX='xlC_r -q32 -qhalt=e '
CMK_LD="xlc_r -q32 -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000 "
CMK_LDXX="xlC_r -q32 -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000"
CMK_CXXPP='xlC -q32 -E'
CMK_LIBS='-lckqt -lhC'
CMK_LD_SHARED='-G'
CMK_QT='aix-light'
CMK_XIOPTS=''
CMK_RANLIB='ranlib'
CMK_AR="ar -X 32 cq"

CMK_NATIVE_CXX='xlC -D_H_UNISTD -DYY_NEVER_INTERACTIVE=1 '
CMK_NATIVE_LDXX='xlC'

CMK_CF77='xlf77_r'
CMK_CF90='xlf90_r -qsuffix=f=f90'
CMK_F90LIBS='-lxlf90_r -lhC'
