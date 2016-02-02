BGQ_TYPICAL_FLOOR=/bgsys/drivers/ppcfloor

# if no floor set, use typical floor path
if test -z "$BGQ_FLOOR"
then
  BGQ_FLOOR=$BGQ_TYPICAL_FLOOR
fi

# if no install path (for experimental) set, use floor
if test -z "$BGQ_INSTALL"
then
  BGQ_INSTALL=$BGQ_TYPICAL_FLOOR
fi

BGQ_ZLIB=/soft/libraries/alcf/current/xl/ZLIB/
BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
if test -d "$BGQ_INSTALL/comm/include"
then
  BGQ_INC="-I$BGQ_INSTALL/comm/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk -I$BGQ_ZLIB/include"
  BGQ_LIB="-L$BGQ_INSTALL/comm/lib -lpami-gcc -L$BGQ_INSTALL/spi/lib -L$BGQ_ZLIB/lib -lSPI -lSPI_cnk -lpthread -lrt"
else
  BGQ_INC="-I$BGQ_INSTALL/comm/sys-fast/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk -I$BGQ_ZLIB/include"
  BGQ_LIB="-L$BGQ_INSTALL/comm/sys-fast/lib -lpami -L$BGQ_INSTALL/spi/lib -L$BGQ_ZLIB/lib -lSPI -lSPI_cnk -lpthread -lrt"
fi

# test if compiler binary present
if test ! -x $BGQ_BIN/powerpc64-bgq-linux-g++
then
 echo "ERROR: Invalid BGQ_INSTALL or BGQ_FLOOR, C/C++ compiler missing"
 exit 1
fi

GCC_OPTS="-Wno-deprecated "
OPTS_CPP="$OPTS_CPP"
OPTS_LD="$OPTS_LD"

CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'
CMK_ENABLE_C11='-qlanglvl=extc1x'
CMK_ENABLE_CPP11='-qlanglvl=extended0x'

CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="$BGQ_BIN/powerpc64-bgq-linux-cpp -E "
CMK_CXX="bgxlC_r -qhalt=e -qnokeyword=__int128 -qtls=local-exec"
CMK_CC="bgxlc_r -qcpluscmt -qhalt=e -qnokeyword=__int128 -qtls=local-exec"
CMK_CXXPP="$BGQ_BIN/powerpc64-bgq-linux-g++ -E "
CMK_GCXX="$BGQ_BIN/powerpc64-bgq-linux-g++ $GCC_OPTS "
CMK_CF77="bgxlf_r "
CMK_CF90="bgxlf90_r  -qsuffix=f=f90"
CMK_CF90_FIXED="bgxlf90_r "

CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'

CMK_RANLIB="$BGQ_BIN/powerpc64-bgq-linux-ranlib "
CMK_AR="$BGQ_BIN/powerpc64-bgq-linux-ar q "
CMK_NM='nm '
CMK_QT="aix"

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_LIBS='-lckqt'
CMK_SYSINC="$BGQ_INC"
CMK_SYSLIBS="$BGQ_LIB"
CMK_F90LIBS="-lxlf90 -lxlopt -lxl -lxlfmath"
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
