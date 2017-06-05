CMK_CC="$CMK_CC -DCMK_GFORTRAN"
CMK_CXX="$CMK_CXX -DCMK_GFORTRAN"

if test -n "$CMK_MACOSX"
then
CMK_F90FLAGS="$CMK_F90FLAGS -fno-common"
CMK_F77FLAGS="$CMK_F90FLAGS -fno-common"
fi

CMK_FPP="$CMK_CPP_C -P -CC"

# Find common gfortran binary names, and choose the last one found
# (presumably the most modern).
CMK_CF90=$(which gfortran gfortran-{4..19} gfortran-mp-{4..19} 2>/dev/null | tail -1)

[ -z $CMK_CF90 ] && { echo 'No gfortran found, exiting'; exit 1; }

# Find libgfortran, which we need to link to manually as the C++ compiler does
# the linking.
f90libdir=$(dirname $($CMK_CF90 -print-file-name=libgfortran.a))
gccehlibdir=$(dirname $($CMK_CF90 -print-file-name=libgcc_eh.a))

CMK_CF90="$CMK_CF90 $CMK_F90FLAGS -fPIC -fno-second-underscore -fdollar-ok "
CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "

CMK_F90LIBS="-L$f90libdir/ -L$gccehlibdir/ -lgfortran -lgcc_eh "

CMK_CF77=$CMK_CF90
CMK_F77LIBS=$CMK_F90LIBS

CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
