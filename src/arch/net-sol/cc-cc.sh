CMK_CPP_CHARM='/usr/ccs/lib/cpp '
CMK_CPP_C='cc -E '
CMK_CC='cc -KPIC '
CMK_CC_RELIABLE='cc '
CMK_CC_FASTEST='cc '
CMK_CXX='CC -KPIC -library=Cstd -instances=global '
CMK_CXXPP='CC -E '
CMK_C_DEBUG='-g'
CMK_C_OPTIMIZE='-fast'
CMK_CXX_DEBUG='-g'
CMK_CXX_OPTIMIZE='-fast'
CMK_LD='cc '
CMK_LDXX='CC -library=Cstd -instances=global '
CMK_LIBS=" -lnsl -lsocket $CHARMLIB/libckqt.a"
CMK_CPP_SUFFIX="cc"
CMK_XLATPP='charmxlat++ -w -p '

CMK_AR='CC -xar -o'
CMK_QT='generic_alloca'

# native compiler
CMK_NATIVE_CC='cc'
CMK_NATIVE_LD='cc'
CMK_NATIVE_CXX='CC -library=Cstd'
CMK_NATIVE_LDXX='CC -library=Cstd'
CMK_NATIVE_LIBS='-lnsl -lsocket'

# Sun Forte Developer 7 f90 7.0
CMK_CF77='f77 -stackvar '
CMK_CF90='f90 -stackvar '
CMK_CF90_FIXED="$CMK_CF90 -fixed "
#CMK_F90LIBS='-lfsu -lsunmath -lfsumai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfui -lfai'
CMK_F90LIBS="-lfui -lfai -lfai2 -lfsumai -lfprodai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfsu -lsunmath -lm -lc -lrt"

