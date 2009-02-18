#CELL_SDK_DIR="/opt/IBM/cell-sdk-1.1"
#CELL_SDK_DIR="/opt/ibm/cell-sdk/prototype"

CMK_CPP_CHARM='/usr/lib/cpp'
CMK_CPP_C='ppu32-gcc -E'
CMK_CXXPP='ppu32-g++ -E'
#CMK_CC="ppu32-gcc -fPIC -w -I$CELL_SDK_DIR/sysroot/usr/include -L$CELL_SDK_DIR/sysroot/usr/lib"
CMK_CC="ppu32-gcc -fPIC -w"
#CMK_CXX="ppu32-g++ -fPIC -w  -I$CELL_SDK_DIR/sysroot/usr/include -L$CELL_SDK_DIR/sysroot/usr/lib"
CMK_CXX="ppu32-g++ -fPIC -w"
#CMK_LD="$CMK_CC -L$CELL_SDK_DIR/sysroot/usr/lib"
CMK_LD="$CMK_CC"
#CMK_LDXX="$CMK_CXX -L$CELL_SDK_DIR/sysroot/usr/lib"
CMK_LDXX="$CMK_CXX"

CMK_SEQ_CC=$CMK_CC
CMK_SEQ_CXX=$CMK_CXX
CMK_SEQ_LD=$CMK_LD
CMK_SEQ_LDXX=$CMK_LDXX

#CMK_SPE_CC="spu-gcc -W -Wall -Winline -Wno-main -Wl,-N -I$CELL_SDK_DIR/sysroot/usr/spu/include -L$CELL_SDK_DIR/sysroot/usr/spu/lib"
CMK_SPE_CC="spu-gcc -W -Wall -Winline -Wno-main -Wl,-N"
#CMK_SPE_CXX="spu-g++ -W -Wall -Winline -Wno-main -Wl,-N -I$CELL_SDK_DIR/sysroot/usr/spu/include -L$CELL_SDK_DIR/sysroot/usr/spu/lib"
CMK_SPE_CXX="spu-g++ -W -Wall -Winline -Wno-main -Wl,-N"
#CMK_SPE_LD="spu-gcc -Wl,-N -L$CELL_SDK_DIR/sysroot/usr/spu/lib"
CMK_SPE_LD="spu-gcc -Wl,-N"
#CMK_SPE_LDXX="spu-g++ -Wl,-N -L$CELL_SDK_DIR/sysroot/usr/spu/lib"
CMK_SPE_LDXX="spu-g++ -Wl,-N"
#CMK_SPERT_LIBS="-lcellspu $CELL_SDK_DIR/sysroot/usr/spu/lib/libc.a $CELL_SDK_DIR/sysroot/usr/spu/lib/libsim.a"
#CMK_SPERT_LIBS="-lcellspu $CELL_SDK_DIR/sysroot/usr/spu/lib/libsim.a"
CMK_SPERT_LIBS="-lcellspu"
CMK_SPE_AR='spu-ar'
CMK_PPU_EMBEDSPU='ppu32-embedspu'

CMK_RANLIB='ppu-ranlib'
CMK_AR='ppu-ar -r'
#CMK_LIBS='-lckqt -lcellppu -lspe'
CMK_LIBS='-lckqt -lcellppu -lspe2'
CMK_LD_SHARED='-shared'

CMK_NATIVE_CC='gcc'
CMK_NATIVE_LD='gcc'
CMK_NATIVE_CXX='g++'
CMK_NATIVE_LDXX='g++'

CMK_CF77='xlf77_r'
CMK_CF90='xlf90_r -qsuffix=f=f90'
CMK_QT='aix32-gcc'
CMK_F90LIBS=''

## DMK - Accel Support for this architecture (CBEA, aka "Cell")
CMK_CELL=1
CMK_PPU_CC="ppu32-gcc -fPIC -w $OPTS"
CMK_PPU_CXX="ppu32-g++ -fPIC -w $OPTS"
CMK_SPU_CC="spu-gcc -W -Winline -Wno-main $OPTS"
CMK_SPU_CXX="spu-g++ -W -Winline $OPTS"
CMK_SPU_LD="spu-gcc -Wl,-N"
CMK_SPU_LDXX="spu-g++ -Wl,-N"
CMK_SPU_AR="spu-ar"
CMK_PPU_EMBEDSPU="ppu32-embedspu"
CMK_SPERT_LIBS="-lcellspu"
