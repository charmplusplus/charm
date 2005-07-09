CMK_CPP_C='icc -E '
CMK_CC="icc -fpic -cxxlib-icc "
CMK_CXX="icpc -fpic -cxxlib-icc "
CMK_CXXPP='icpc -E -cxxlib-icc '
CMK_LD='icc -i_dynamic -cxxlib-icc '
CMK_LDXX='icpc -i_dynamic -cxxlib-icc '
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC="$CMK_CC"
CMK_NATIVE_CXX="$CMK_CXX"
CMK_NATIVE_LD="$CMK_LD"
CMK_NATIVE_LDXX="$CMK_LDXX"

# for absoft?
#CMK_F90LIBS='-L/usr/local/intel/compiler70/ia32/lib -L/opt/intel/compiler70/ia32/lib -lintrins -lIEPCF90 -lPEPCF90 -lF90 -lintrins -limf  '
#CMK_MOD_NAME_ALLCAPS=1
#CMK_MOD_EXT="mod"
#CMK_F90_USE_MODDIR=""

# for Intel Fortran compiler ifc
CMK_CF77="ifc -auto "
CMK_CF90="ifc -auto "
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
F90DIR=`which ifc 2> /dev/null`
if test -x "$F90DIR" 
then
  F90LIBDIR="`dirname $F90DIR`/../lib"
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
CMK_F77LIBS="$CMK_F90LIBS"
CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=""
