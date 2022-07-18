PAMI_INC=${MPI_ROOT}/pami_devel/include
PAMI_LIB=${MPI_ROOT}/lib/pami_port

LIBCOLL_INC=${MPI_ROOT}/pami_devel/include
LIBCOLL_LIB=${MPI_ROOT}/lib/pami_port

CXX=xlC_r
CC=xlc_r

CMK_CPP_CHARM='cpp -P'
CMK_CPP_C="$CC"
CMK_CC="$CC "
CMK_CXX="$CXX "
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX "

CMK_CPP_C_FLAGS="-E"

CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_SYSINC="-I$PAMI_INC -I$LIBCOLL_INC"
#CMK_SYSLIBS="-L$PAMI_LIB -L/usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lnuma -lstdc++ -lc -ldl -lrt -lpthread"
CMK_SYSLIBS="-L$PAMI_LIB -L$LIBCOLL_LIB -lcollectives -L/usr/lib/powerpc64le-linux-gnu -lpami -libverbs -lstdc++ -lc -ldl -lrt -lpthread"

CMK_NATIVE_LIBS=''

CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT='mod'
CMK_F90_MODINC='-p'
CMK_F90_USE_MODDIR=''

CMK_COMPILER='xlc_r'
