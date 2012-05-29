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

BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
BGQ_INC="-I$BGQ_INSTALL/comm/sys/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk/"

BGQ_LIB="-L$BGQ_INSTALL/comm/sys-fast/lib -lpami -L$BGQ_INSTALL/spi/lib -lSPI -lSPI_cnk -lpthread -lrt" 
#"-pg -L/bghome/boger/sandbox/src-110606/bgq/work/gnu-linux/powerpc64-bgq-linux/lib -lc"

# test if compiler binary present
if test ! -x $BGQ_BIN/powerpc64-bgq-linux-g++
then
 echo "ERROR: Invalid BGQ_INSTALL or BGQ_FLOOR, C/C++ compiler missing"
 exit 1
fi

OPTS_CPP="$OPTS_CPP"
GCC_OPTS="-Wno-deprecated $BGQ_INC"
OPTS_LD="$OPTS_LD"

CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="$BGQ_BIN/powerpc64-bgq-linux-cpp -E "
CMK_CXX="$BGQ_BIN/powerpc64-bgq-linux-g++ $GCC_OPTS "
CMK_GCXX="$BGQ_BIN/powerpc64-bgq-linux-g++ $GCC_OPTS "
CMK_CC="$BGQ_BIN/powerpc64-bgq-linux-gcc $GCC_OPTS "
CMK_CXXPP="$BGQ_BIN/powerpc64-bgq-linux-g++ -E "
CMK_CF77="$BGQ_BIN/powerpc64-bgq-linux-gfortran "
CMK_CF90='f90'
CMK_RANLIB="$BGQ_BIN/powerpc64-bgq-linux-ranlib "
CMK_AR="$BGQ_BIN/powerpc64-bgq-linux-ar q "
CMK_SYSLIBS="$BGQ_LIB"
CMK_LIBS='-lckqt'
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
#CMK_SEQ_LIBS=''
#CMK_SEQ_CC="$BGQ_BIN/powerpc64-bgq-linux-gcc -Wno-deprecated "
#CMK_SEQ_LD="$CMK_SEQ_CC"
#CMK_SEQ_CXX="$BGQ_BIN/powerpc64-bgq-linux-g++ -Wno-deprecated "
#CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS='-lf90math -lfio -lU77 -lf77math '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"
CMK_QT="generic64"
