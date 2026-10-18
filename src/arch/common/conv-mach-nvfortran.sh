COMMENT="Use nvfortran fortran compiler in $NVHPC_DIR"
NVHPC_DIR=`command -v nvfortran`
if test "$NVHPC_DIR" = ''
then
  echo charmc> Fatal error: nvfortran not found!
  exit 1
fi
NVHPC_DIR="`dirname $NVHPC_DIR`/.."
CMK_CF77="$NVHPC_DIR/bin/nvfortran "
CMK_CF90="$NVHPC_DIR/bin/nvfortran "
CMK_CF90_FIXED="$CMK_CF90 -Mfixed "
CMK_F90MAINLIBS="$NVHPC_DIR/lib/f90main.o"
CMK_F90LIBS="-L$NVHPC_DIR/lib -lnvf -lm -lnvc "
CMK_F90_MODINC="-module"
CMK_F90_USE_MODDIR=""
