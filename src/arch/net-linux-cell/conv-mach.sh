CMK_CPP_CHARM='/usr/lib/cpp'
CMK_CPP_C='ppu32-gcc -E'
CMK_CXXPP='ppu32-g++ -E'
CMK_CC='ppu32-gcc -fPIC -w '
CMK_CXX='ppu32-g++ -fPIC -w '
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_RANLIB='ppu-ranlib'
CMK_AR='ppu-ar -r'
CMK_LIBS='-lckqt'
CMK_LD_SHARED='-shared'

CMK_SEQ_CC='ppu32-gcc -fPIC '
CMK_SEQ_CXX='ppu32-g++ -fPIC '

CMK_NATIVE_CC='gcc'
CMK_NATIVE_LD='gcc'
CMK_NATIVE_CXX='g++'
CMK_NATIVE_LDXX='g++'

CMK_CF77='xlf77_r'
CMK_CF90='xlf90_r -qsuffix=f=f90'
CMK_QT='aix32-gcc'
CMK_XIOPTS=''
CMK_F90LIBS=''
CMK_MOD_EXT=''
