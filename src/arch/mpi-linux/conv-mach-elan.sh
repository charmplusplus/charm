CMK_CPP_C="icc -E"
CMK_CC="icc "
CMK_CXX="icpc "
CMK_CXXPP="icc -E "
OPTS_CPP="$OPTS_CPP -I/usr/lib/mpi/include"
CMK_LD="icc "
CMK_LDXX="icpc "
CMK_LIBS="-lckqt -L/usr/lib/mpi/lib -L/scratch/release/head/quadrics/lib/Linux_i686 -lmpi -lelan "
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_CF77="ifc "
CMK_CF90="ifc "
CMK_CF90_FIXED="ifc "
CMK_F90LIBS=' -lmpifarg -lifport -lifcore '
CMK_F90_USE_MODDIR=
CMK_F90_MODINC=""
