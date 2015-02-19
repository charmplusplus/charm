BGQ_ZLIB=/soft/libraries/alcf/current/gcc/ZLIB/

BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
if test -d "$BGQ_INSTALL/comm/lib"
then
  BGQ_INC="-I$BGQ_INSTALL/comm/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk -I$BGQ_ZLIB/include"
  BGQ_LIB="-L$BGQ_INSTALL/comm/lib -lpami-gcc -L$BGQ_INSTALL/spi/lib -L$BGQ_ZLIB/lib -lSPI -lSPI_cnk -lpthread -lrt"
else
  BGQ_INC="-I$BGQ_INSTALL/comm/sys-fast/include -I$BGQ_INSTALL/spi/include -I$BGQ_INSTALL -I$BGQ_INSTALL/spi/include/kernel/cnk -I$BGQ_ZLIB/include"
  BGQ_LIB="-L$BGQ_INSTALL/comm/sys-fast/lib -lpami -L$BGQ_INSTALL/spi/lib -L$BGQ_ZLIB/lib -lSPI -lSPI_cnk -lpthread -lrt"
fi

CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="$BGQ_BIN/powerpc64-bgq-linux-cpp -E "
CMK_CXX="$BGQ_BIN/powerpc64-bgq-linux-g++ $GCC_OPTS "
CMK_GCXX="$BGQ_BIN/powerpc64-bgq-linux-g++ $GCC_OPTS "
CMK_CC="$BGQ_BIN/powerpc64-bgq-linux-gcc $GCC_OPTS "
CMK_CXXPP="$BGQ_BIN/powerpc64-bgq-linux-g++ -E "
CMK_CF77="$BGQ_BIN/powerpc64-bgq-linux-gfortran "
CMK_CF90="$BGQ_BIN/powerpc64-bgq-linux-gfortran "
CMK_RANLIB="$BGQ_BIN/powerpc64-bgq-linux-ranlib "
CMK_AR="$BGQ_BIN/powerpc64-bgq-linux-ar q "
CMK_SYSINC="$BGQ_INC" 
CMK_SYSLIBS="$BGQ_LIB"
CMK_LIBS='-lckqt'
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS='-lgfortran '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_QT="generic64-light"
