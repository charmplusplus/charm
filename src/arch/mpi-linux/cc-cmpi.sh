# for ChaMPIon/Pro 1.1.1
#
# setting CMK_LD to use cmpic++ to avoid libcprts.so.5 mixed gnu/intel libs

CMK_CPP_C="cmpicc -E"
CMK_CC="cmpicc -icc -gm -fPIC "
CMK_CXX="cmpic++ -icc -gm -fPIC "
CMK_CXXPP="cmpi++ -E "
CMK_LD="cmpicc -icc -fPIC -cxxlib-gcc "
CMK_LDXX="cmpic++ -icc -fPIC "
CMK_LIBS="-lckqt -lcmpi "

CMK_SEQ_CC="icc -fPIC "
CMK_SEQ_CXX="icpc -fPIC "
CMK_SEQ_LD="icpc  -fPIC -cxxlib-gcc "
CMK_SEQ_LDXX="icpc  -fPIC -cxxlib-gcc "

# fortran compiler
CMK_CF77="cmpifc -ifc "
CMK_CF90="cmpif90c -ifc "
CMK_CF90_FIXED="cmpif90c -ifc "
CMK_F90LIBS='-lifcore  '
CMK_F77LIBS=$CMK_F90LIBS
CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=
CMK_F90_MODINC=""

