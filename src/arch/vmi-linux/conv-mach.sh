VMI_DIR="/usr/local/vmi-2.1.0-1-gcc"
#
VMI_INCDIR="-I$VMI_DIR/include" 
VMI_LIBDIR="-L$VMI_DIR/lib"
#
CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="gcc -E -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CC="gcc -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CXX="g++ -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CXXPP="$CMK_CC -x c++ -E "
CMK_CF77="f77"
CMK_CF90="f90"
#CMK_LD="$CMK_CC -rdynamic -pthread $VMI_LIBDIR "
CMK_LD="$CMK_CC -rdynamic -pthread -Wl,-rpath,/usr/local/vmi-2.1.0-1-gcc/lib $VMI_LIBDIR "
#CMK_LDXX="$CMK_CXX -rdynamic -pthread $VMI_LIBDIR"
CMK_LDXX="$CMK_CXX -rdynamic -pthread -Wl,-rpath,/usr/local/vmi-2.1.0-1-gcc/lib $VMI_LIBDIR"
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lvmi20 -lcurl -ldl -lexpat -lssl -lcrypto'
CMK_QT='generic'
CMK_XIOPTS=''
CMK_F90LIBS='-lvast90 -lg2c'
CMK_MOD_EXT="vo"
