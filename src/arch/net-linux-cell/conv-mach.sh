CELL_SDK_DIR="$HOME/cellsim"
CMK_CPP_CHARM='/usr/lib/cpp'
CMK_CPP_C='ppu32-gcc -E'
CMK_CXXPP='ppu32-g++ -E'
CMK_CC="ppu32-gcc -fPIC -w -I$CELL_SDK_DIR/sysroot/usr/include"
CMK_CXX="ppu32-g++ -fPIC -w  -I$CELL_SDK_DIR/sysroot/usr/include"
CMK_LD="$CMK_CC -L$HOME/cellsim/systemsim-cell-release/run/cell/linux/spert -L$CELL_SDK_DIR/sysroot/usr/lib "
CMK_LDXX="$CMK_CXX -L$HOME/cellsim/systemsim-cell-release/run/cell/linux/spert -L$CELL_SDK_DIR/sysroot/usr/lib "

CMK_SEQ_CC=$CMK_CC
CMK_SEQ_CXX=$CMK_CXX
CMK_SEQ_LD=$CMK_LD
CMK_SEQ_LDXX=$CMK_LDXX

CMK_RANLIB='ppu-ranlib'
CMK_AR='ppu-ar -r'
CMK_LIBS='-lckqt -lcellppu -lspe'
CMK_LD_SHARED='-shared'

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
