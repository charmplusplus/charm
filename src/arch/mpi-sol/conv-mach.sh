CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C='mpicc -E'
CMK_CC='mpicc '
CMK_CXX='mpiCC '
CMK_CXXPP='mpiCC -E '
CMK_QT='solaris-cc'
CMK_LIBS='-lckqt -lmpich -lsocket -lnsl -lthread '

# shared library
CMK_LD_SHARED="-G"
CMK_LD_LIBRARY_PATH="-R $CHARMLIBSO/"

# native compiler
CMK_NATIVE_LIBS=
CMK_NATIVE_CC='cc'
CMK_NATIVE_LD='cc'
CMK_NATIVE_CXX='CC'
CMK_NATIVE_LDXX='CC'

# for Sun Forte Developer 7 f90 7.0
# avoid -C (array boundry checking flag which is slow)
CMK_CF77='f77 -stackvar '
CMK_CF90='f90 -stackvar '
CMK_CF90_FIXED="$CMK_CF90 -fixed "
#CMK_F90LIBS="-L/opt/SUNWspro/lib -lfsu -lsunmath -lfsumai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfui -lfai"
CMK_F90LIBS="-L/opt/SUNWspro/lib -lfui -lfai -lfai2 -lfsumai -lfprodai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfsu -lsunmath -lm -lc -lrt"

CMK_AR='CC -xar -o'
CMK_RANLIB='ranlib'
CMK_XIOPTS=''

