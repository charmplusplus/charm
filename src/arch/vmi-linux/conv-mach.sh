VMI_INCDIR="-I/home/koenig/THESIS/VMI20-install/include" 
VMI_LIBDIR="-L/home/koenig/THESIS/VMI20-install/lib"
CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="gcc3 -E $CMK_INCDIR $VMI_INCDIR "
CMK_CC="gcc3 $CMK_INCDIR $VMI_INCDIR "
CMK_CXX="g++3 $CMK_INCDIR $VMI_INCDIR "
CMK_CXXPP="$CMK_CC -x c++ -E "
CMK_CF77="f77"
CMK_CF90="f90"
CMK_LD="$CMK_CC -rdynamic -pthread $VMI_LIBDIR "
CMK_LDXX="$CMK_CXX -rdynamic -pthread $VMI_LIBDIR"
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lvmi20 -lcurl -ldl -lexpat -lssl -lcrypto'
CMK_QT='generic64'
CMK_XIOPTS=''
CMK_F90LIBS='-lvast90 -lg2c'
CMK_MOD_EXT="vo"



######################################################################
##CMK_CC='gcc '
#CMK_CC_RELIABLE='gcc '
#CMK_CC_FASTEST='gcc '
#CMK_C_DEBUG='-g'
#CMK_C_OPTIMIZE='-O'
#CMK_CXX_DEBUG='-g'
#CMK_CXX_OPTIMIZE='-O'
##CMK_LD='gcc -static '
##CMK_LDXX='g++ -static '
#CMK_LD77=''
#CMK_M4='m4'
#CMK_SUF='o'
#CMK_AR='ar q'
##CMK_LIBS='-lckqt'
#CMK_SEQ_LIBS=''
#CMK_SEQ_CC='gcc'
#CMK_SEQ_LD='gcc'
#CMK_SEQ_CXX='g++'
#CMK_SEQ_LDXX='g++'
#CMK_CPP_SUFFIX="ii"
#CMK_XLATPP='charmxlat++ '

