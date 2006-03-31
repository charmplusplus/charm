
# for Intel Fortran compiler ifc
CMK_CF77="ifc -auto "
CMK_CF90="ifc -auto "
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
F90DIR=`which ifc 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
if test -x "$F90DIR" 
then
  F90LIBDIR="`dirname $F90DIR`/../lib"
  F90MAIN="$F90LIBDIR/for_main.o"
  if test -f $F90LIBDIR/libifcore.a
  then
    CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
  else
    CMK_F90LIBS="-L$F90LIBDIR -lintrins -lIEPCF90 -lPEPCF90 -lF90 -lintrins -limf "
  fi
fi
CMK_F77LIBS="$CMK_F90LIBS"

# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "

CMK_F90_USE_MODDIR=""
