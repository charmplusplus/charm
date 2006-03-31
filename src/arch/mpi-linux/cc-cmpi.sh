# for ChaMPIon/Pro 1.1.1
#
# setting CMK_LD to use cmpic++ to avoid libcprts.so.5 mixed gnu/intel libs

CMK_CPP_C="cmpicc -E"
CMK_CC="cmpicc -icc -gm -fPIC "
CMK_CXX="cmpic++ -icc -gm -fPIC "
CMK_CXXPP="cmpi++ -E "
CMK_LD="cmpicc -icc -fPIC -cxxlib-gcc "
CMK_LDXX="cmpic++ -ccl icpc -fPIC "
CMK_LIBS="-lckqt -lcmpi "

#CMK_SEQ_CC="icc -fPIC "
#CMK_SEQ_CXX="icpc -fPIC "
#CMK_SEQ_LD="icpc  -fPIC -cxxlib-gcc "
#CMK_SEQ_LDXX="icpc  -fPIC -cxxlib-gcc "
CMK_SEQ_CC="$CMK_CC"
CMK_SEQ_CXX="$CMK_CXX "
CMK_SEQ_LD="$CMK_LD "
CMK_SEQ_LDXX="$CMK_LDXX "

# fortran compiler
CMK_CF77="cmpifc -ifc -auto "
CMK_CF90="cmpif90c -ifc -auto "
CMK_CF90_FIXED="cmpif90c -ifc -auto "
CMK_F90LIBS="-lifcore -lifport $F90MAIN "
# for_main.o is important for main() in f90 code
F90DIR=`which ifort 2> /dev/null`
test -x "$F90DIR" && F90MAIN="`dirname $F90DIR`/../lib/for_main.o"
CMK_F90MAINLIBS="$F90MAIN "
CMK_F77LIBS="$CMK_F90LIBS"
CMK_F90_USE_MODDIR=
CMK_F90_MODINC=""

