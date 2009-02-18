CMK_CPP_C='icc -E '
CMK_DEPRECATED="-cxxlib-icc"
CMK_CC="icc -fpic "
CMK_CXX="icpc -fpic "
CMK_CXXPP='icpc -E '
CMK_LD='icc -i_dynamic '
CMK_LDXX='icpc -i_dynamic '
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC="$CMK_CC"
CMK_NATIVE_CXX="$CMK_CXX"
CMK_NATIVE_LD="$CMK_LD"
CMK_NATIVE_LDXX="$CMK_LDXX"
CPPFLAGS="$CPPFLAGS -fpic "
LDFLAGS="$LDFLAGS -i_dynamic "

# for absoft?
#CMK_F90LIBS='-L/usr/local/intel/compiler70/ia32/lib -L/opt/intel/compiler70/ia32/lib -lintrins -lIEPCF90 -lPEPCF90 -lF90 -lintrins -limf '
#CMK_MOD_NAME_ALLCAPS=1

# for Intel Fortran compiler 8.0 and higher which is renamed to ifort from ifc
# does not work for ifc 7.0
CMK_CF77="ifort -auto "
CMK_CF90="ifort -auto "
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
F90DIR=`which ifort 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
if test -x "$F90DIR" 
then
  F90LIBDIR="`dirname $F90DIR`/../lib"
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
CMK_F77LIBS="$CMK_F90LIBS"

CMK_F90_USE_MODDIR=""
