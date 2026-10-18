# Intel LLVM-based compiler Intel® Fortran Compiler
# https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-ifort-to-ifx.html


CMK_FPP="$CMK_CPP_C -P -CC"

CMK_CF90="ifx"

CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "

CMK_F90LIBS="-lrt"

CMK_CF77=$CMK_CF90
CMK_F77LIBS=$CMK_F90LIBS

CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
