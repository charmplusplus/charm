CMK_DEFS="$CMK_DEFS -q32"
CMK_CC="xlc_r -qcpluscmt $CMK_DEFS "
CMK_CXX="xlC_r $CMK_DEFS"
CMK_C_OPTIMIZE='-O3 -qstrict -Q  '
CMK_CXX_OPTIMIZE='-O3 -qstrict -Q '
CMK_PIC='-qpic=small'

CMK_LD="$CMK_CC -brtl "
CMK_LDXX="$CMK_CXX -brtl "

CMK_NATIVE_CC='xlc_r -q32 '
CMK_NATIVE_LD='xlc_r -q32 '
CMK_NATIVE_CXX='xlC_r -q32 -D_H_UNISTD -DYY_NEVER_INTERACTIVE=1 '
CMK_NATIVE_LDXX='xlC_r -q32 '

CMK_CF77='xlf_r -q32  '
CMK_CF90='xlf90_r  -qsuffix=f=f90 -q32 ' 
CMK_CF90_FIXED='xlf90_r -q32 ' 

if test "$isAIX" = "true"
then
  AR_OPTS="-X 32"
fi
CMK_AR="ar $AR_OPTS cq"
CMK_NM="nm $AR_OPTS"

CMK_C_OPENMP="-qsmp=omp"
