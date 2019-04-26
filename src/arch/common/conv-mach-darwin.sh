CMK_MACOSX=1

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_QT='generic64-light'
CMK_XIOPTS=''

CMK_CC_FLAGS="$CMK_CC_FLAGS -fPIC -dynamic -fno-common "
CMK_LD_FLAGS="$CMK_LD_FLAGS -Wl,-no_pie "
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -fPIC -dynamic -fno-common -stdlib=libc++ "
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -multiply_defined suppress -Wl,-no_pie -stdlib=libc++ "

# setting for shared lib
CMK_SHARED_SUF="dylib"
CMK_LD_SHARED=" -dynamic -dynamiclib -undefined dynamic_lookup "
CMK_LD_SHARED_ABSOLUTE_PATH=true

CMK_DEFS="$CMK_DEFS -mmacosx-version-min=10.7 -D_DARWIN_C_SOURCE"

if command -v gfortran >/dev/null 2>&1
then
  . $CHARMINC/conv-mach-gfortran.sh
fi

# Assumes gfortran compiler:
CMK_CF77="$CMK_CF77 -mmacosx-version-min=10.7"
CMK_CF90="$CMK_CF90 -mmacosx-version-min=10.7"
