VMI_DIR="/usr/local/vmi-2.1.0-1-gcc"
VMI_DIR="/opt/vmi-2.2.0-2-gcc"
#
VMI_INCDIR="-I$VMI_DIR/include" 
VMI_LIBDIR="-L$VMI_DIR/lib"
#
CMK_CPP_CHARM="/lib/cpp -P "
CMK_CPP_C="gcc -E -m64 -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CC="gcc -fPIC -m64 -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CXX="g++ -fPIC -m64 -DNO_LOCK $CMK_INCDIR $VMI_INCDIR "
CMK_CXXPP="$CMK_CC -x c++ -E -m64 -DNO_LOCK "
CMK_LD="$CMK_CC -rdynamic -pthread -Wl,-rpath,$VMI_DIR/lib $VMI_LIBDIR "
CMK_LDXX="$CMK_CXX -rdynamic -pthread -Wl,-rpath,$VMI_DIR/lib $VMI_LIBDIR "
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lvmi20 -lcurl -ldl -lexpat -lssl -lcrypto'
CMK_QT='generic64'
CMK_XIOPTS=''

CMK_CF77="f77 "
CMK_CF90="f90 "
CMK_F90LIBS='-lvast90 -lg2c'
