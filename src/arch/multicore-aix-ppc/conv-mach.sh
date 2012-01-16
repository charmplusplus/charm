CMK_DEFS=' -D_REENTRANT -D_THREAD_SAFE '
CMK_CPP_CHARM='/usr/lib/cpp'
CMK_CPP_C="xlc -E $CMK_DEFS "
CMK_CC="xlc_r -qcpluscmt $CMK_DEFS "
CMK_CXX="xlC_r $CMK_DEFS "
CMK_LD="xlc_r -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000 $CMK_DEFS "
CMK_LDXX="xlC_r -brtl -bmaxdata:0x80000000 -bmaxstack:0x80000000 $CMK_DEFS "
CMK_CXXPP="xlC -E $CMK_DEFS "
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lhC -lpthreads'
CMK_LD_SHARED='-G'
CMK_QT='aix-light'
CMK_XIOPTS=''

CMK_NATIVE_CXX='xlC -D_H_UNISTD -DYY_NEVER_INTERACTIVE=1 '
CMK_NATIVE_LDXX='xlC'

CMK_CF77='xlf77_r'
CMK_CF90='xlf90_r -qsuffix=f=f90'
CMK_F90LIBS='-lxlf90_r -lhC'

