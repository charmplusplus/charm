BGQ_ZLIB=/soft/libraries/alcf/current/xl/ZLIB/
BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
BGQ_INC="-I$BGQ_INSTALL/comm/sys/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk -I$BGQ_ZLIB/include"
BGQ_LIB="-L$BGQ_INSTALL/comm/sys-fast/lib -lpami -L$BGQ_INSTALL/spi/lib -L$BGQ_ZLIB/lib -lSPI -lSPI_cnk -lpthread -lrt" 
CMK_SYSLIBS="$BGQ_LIB"

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

