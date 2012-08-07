# for Intel Fortran compiler 8.0 and higher which is renamed to ifort from ifc
# does not work for ifc 7.0
CMK_CF77="ifort -auto -fpic "
CMK_CF90="ifort -auto -fpic "
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
F90DIR=`which ifort 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
if test -x "$F90DIR" 
then
  F90DIR=`dirname $F90DIR`
  Minor=`basename $F90DIR`
  if test "$Minor" = "intel64"
  then
    F90DIR=`dirname $F90DIR`
    F90LIBDIR="$F90DIR/../lib/$Minor"
  else
    F90LIBDIR="$F90DIR/../lib"
  fi
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
if test -z "$ICC_STATIC"
then
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
else
CMK_F90LIBS="$F90LIBDIR/libifcore.a $F90LIBDIR/libifport.a "
fi
CMK_F77LIBS="$CMK_F90LIBS"

CMK_F90_USE_MODDIR=""
