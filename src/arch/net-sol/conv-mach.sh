CMK_DEFS="-fPIC"
CMK_CPP_CHARM="/usr/local/bin/cpp -P"
CMK_CPP_C="gcc -E $CMK_DEFS"
CMK_CC="gcc $CMK_DEFS"
CMK_CXX="g++ $CMK_DEFS "
CMK_CXXPP="g++ -x c++ -E $CMK_DEFS "
CMK_XIOPTS=''
CMK_LIBS=' -lnsl -lsocket -lckqt -lthread'

# for Sun Forte Developer 7 f90 7.0
# avoid -C (array boundry checking flag)
CMK_CF77='f77 -stackvar '
CMK_CF90='f90 -stackvar '
CMK_CF90_FIXED="$CMK_CF90 -fixed "
#CMK_F90LIBS='-L/opt/SUNWspro/lib -lfsu -lsunmath -lfsumai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfui -lfai'
CMK_F90LIBS="-L/opt/SUNWspro/lib -lfui -lfai -lfai2 -lfsumai -lfprodai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfsu -lsunmath -lm -lc -lrt"
CMK_F90_USE_MODDIR=1

# shared library
CMK_LD_SHARED="-G"
CMK_LD_LIBRARY_PATH="-R $CHARMLIBSO/"

# native compiler
CMK_NATIVE_CC="$CMK_CC"
CMK_NATIVE_LD="$CMK_CC"
CMK_NATIVE_CXX="$CMK_CXX"
CMK_NATIVE_LDXX="$CMK_CXX"
CMK_NATIVE_LIBS='-lnsl -lsocket -lthread '

CMK_RANLIB='true'
CMK_QT='solaris-gcc'
