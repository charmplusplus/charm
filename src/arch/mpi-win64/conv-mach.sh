
if test -n "$CCP_LIB64"
then
  HPC_SDK="$CCP_LIB64\..\.."
else
  HPC_SDK="c:\Program Files\Microsoft MPI"
fi
HPC_SDK=`cygpath -d "$HPC_SDK"`

CMK_CC="$CHARMBIN/unix2nt_cc -D_CRT_SECURE_NO_DEPRECATE -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="$CMK_CC -E"
CMK_CXX="$CHARMBIN/unix2nt_cc -D_CRT_SECURE_NO_DEPRECATE  -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_CXXPP=$CMK_CC
CMK_LD="$CMK_CC -L `cygpath -u "$HPC_SDK\Lib\amd64"` -lmsmpi"
CMK_LDXX="$CMK_CXX -L `cygpath -u "$HPC_SDK\Lib\amd64"` -lmsmpi"

CMK_SEQ_CC="$CMK_CC"
CMK_SEQ_CXX="$CMK_CXX"
CMK_SEQ_LD="$CMK_LD"
CMK_SEQ_LDXX="$CMK_LDXX"

CMK_CF77="f77"
CMK_CF90="f90"
CMK_AR="$CHARMBIN/unix2nt_ar "
CMK_RANLIB="echo "
CMK_LIBS=""
CMK_XIOPTS=""
CMK_F90LIBS="-lvast90 -lg2c"
CMK_MOD_EXT="vo"
CMK_POST_EXE=".exe"
CMK_QT="none"
