CMK_CPP_CHARM='cpp -P'
CMK_CPP_C='mpicc'
CMK_CC='mpicc '
CMK_CXX='mpicxx '

CMK_CPP_C_FLAGS="-E"

# avoid the need to link -lmpi_cxx on some systems
CMK_DEFS="$CMK_DEFS -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX"

CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_F90_USE_MODDIR=1
CMK_F90_MODINC='-p'

CMK_COMPILER='mpicc'
