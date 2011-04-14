BGP_TYPICAL_FLOOR=/bgsys/drivers/ppcfloor

# if no floor set, use typical floor path
if test -z "$BGP_FLOOR"
then
  BGP_FLOOR=$BGP_TYPICAL_FLOOR
fi

# if no install path (for experimental) set, use floor
if test -z "$BGP_INSTALL"
then
  BGP_INSTALL=$BGP_FLOOR
fi

BGP_ZLIB=/soft/apps/zlib-1.2.3

BGP_BIN=$BGP_FLOOR/gnu-linux/bin
BGP_INC="-I$BGP_INSTALL/comm/include -I$BGP_INSTALL/arch/include -I$BGP_ZLIB/include "

BGP_LIB="-L$BGP_INSTALL/comm/lib -L$BGP_INSTALL/runtime/SPI -L$BGP_ZLIB/lib "

# test if compiler binary present
if test ! -x $BGP_BIN/powerpc-bgp-linux-g++
then
 echo "ERROR: Invalid BGP_INSTALL or BGP_FLOOR, C/C++ compiler missing"
 exit 1
fi

OPTS_CPP="$OPTS_CPP"
GCC_OPTS="-gdwarf-2 $BGP_INC"
GXX_OPTS="$GCC_OPTS -Wno-deprecated"
OPTS_LD="$OPTS_LD"

CMK_CPP_CHARM="$BGP_BIN/powerpc-bgp-linux-cpp -P"
CMK_CPP_C="$BGP_BIN/powerpc-bgp-linux-cpp -E "
CMK_CXX="$BGP_BIN/powerpc-bgp-linux-g++ $GXX_OPTS -DMPICH_IGNORE_CXX_SEEK "
CMK_GCXX="$BGP_BIN/powerpc-bgp-linux-g++ $GXX_OPTS "
CMK_CC="$BGP_BIN/powerpc-bgp-linux-gcc $GCC_OPTS "
CMK_CXXPP="$BGP_BIN/powerpc-bgp-linux-g++ -E "
CMK_CF77="$BGP_BIN/powerpc-bgp-linux-gfortran "
CMK_CF90="$BGP_BIN/powerpc-bgp-linux-gfortran "
CMK_RANLIB="$BGP_BIN/powerpc-bgp-linux-ranlib "
CMK_AR="$BGP_BIN/powerpc-bgp-linux-ar q "
CMK_LD="$CMK_CC $BGP_LIB"
CMK_LDXX="$CMK_CXX $BGP_LIB"
CMK_LIBS='-lckqt -ldcmf-fast.cnk -ldcmfcoll.cnk -lpthread -lrt -lSPI-fast.cna'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
#CMK_SEQ_LIBS=''
#CMK_SEQ_CC="$BGP_BIN/powerpc-bgp-linux-gcc -Wno-deprecated "
#CMK_SEQ_LD="$CMK_SEQ_CC"
#CMK_SEQ_CXX="$BGP_BIN/powerpc-bgp-linux-g++ -Wno-deprecated "
#CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS='-lgfortran '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_QT="generic-light"
CMK_PRODUCTION='-DOPTIMIZED_MULTICAST=1 -DOPT_RZV '
