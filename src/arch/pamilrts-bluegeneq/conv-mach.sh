. $CHARMINC/cc-bluegene.sh

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

CMK_CXX="bgxlC_r -qhalt=e -qnokeyword=__int128 -qtls=local-exec -DCMK_USING_XLC=1"
CMK_CC="bgxlc_r -qcpluscmt -qhalt=e -qnokeyword=__int128 -qtls=local-exec"

CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_CF77="bgxlf_r "
CMK_CF90="bgxlf90_r  -qsuffix=f=f90"
CMK_CF90_FIXED="bgxlf90_r "

CMK_SYSINC="$BGQ_INC"
CMK_SYSLIBS="$BGQ_LIB"
