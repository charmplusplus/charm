CMK_MACOSX=1

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_QT='generic64-light'
CMK_XIOPTS=''

CMK_CC_FLAGS="$CMK_CC_FLAGS -fPIC -dynamic -fno-common "
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -fPIC -dynamic -fno-common -stdlib=libc++ "
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -framework Foundation -framework IOKit -multiply_defined suppress -stdlib=libc++ "

# setting for shared lib
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_ABSOLUTE_PATH=true

CMK_DEFS="$CMK_DEFS -D_DARWIN_C_SOURCE"

if command -v gfortran gfortran-{19..4} gfortran-mp-{19..4} >/dev/null 2>&1
then
  . $CHARMINC/conv-mach-gfortran.sh
fi

CMK_NATIVE_CC='clang'
CMK_NATIVE_LD='clang'
CMK_NATIVE_CXX='clang++'
CMK_NATIVE_LDXX='clang++'

CMK_NATIVE_CC_FLAGS="$CMK_CC_FLAGS"
CMK_NATIVE_LD_FLAGS="$CMK_LD_FLAGS"
CMK_NATIVE_CXX_FLAGS="$CMK_CXX_FLAGS"
CMK_NATIVE_LDXX_FLAGS="$CMK_LDXX_FLAGS"

# Apple clang still defaults to C++98; Charm++ requires C++17. CMK_SEQ_CXX_FLAGS
# is left alone: conv-config.sh defaults it from CMK_CXX_FLAGS when unset, so
# setting it here would only strip the other flags that default carries.
# CMK_NATIVE_CXX_FLAGS is snapshotted above before per-target conv-mach.sh
# scripts run (charmxi et al. build with NATIVE), so it needs an explicit append.
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -std=gnu++17"
CMK_NATIVE_CXX_FLAGS="$CMK_NATIVE_CXX_FLAGS -std=gnu++17"
