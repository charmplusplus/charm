CMK_DEFS=' -I.  -DCMK_FIND_FIRST_OF_PREDICATE=1 '
CMK_CPP_CHARM="/usr/ccs/lib/cpp $CMK_DEFS"
CMK_CPP_C="gcc -E $CMK_DEFS"
CMK_CC="cc -m64 $CMK_DEFS"
CMK_CXX="CC -m64 $CMK_DEFS -features=zla "
CMK_CXXPP="CC -m64 -E $CMK_DEFS "
CMK_CF77="f77 -m64 "
CMK_CF90="f90 -m64 "
CMK_RANLIB='true'
CMK_LIBS=' -lnsl -lsocket -lckqt'
CMK_NATIVE_LIBS=' -lnsl -lsocket'
CMK_XIOPTS=''

# shared library
CMK_LD_SHARED="-G"
CMK_LD_LIBRARY_PATH="-R $CHARMLIBSO/"

CMK_QT='generic64-light'
#CMK_QT='generic'
