
if test -n "$CCP_LIB64"
then
  HPC_SDK="$CCP_LIB64\..\.."
else
  HPC_SDK="c:\Program Files\Microsoft MPI"
fi
HPC_SDK=`cygpath -d "$HPC_SDK"`

CMK_CC="$CHARMBIN/unix2nt_cc"
CMK_CPP_CHARM="/usr/bin/cpp -P"
CMK_CPP_C="$CMK_CC"
CMK_CXX="$CHARMBIN/unix2nt_cc"
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_CC_FLAGS="-D_CRT_SECURE_NO_DEPRECATE -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_CPP_C_FLAGS="$CMK_CC_FLAGS -E"
CMK_CXX_FLAGS="-D_CRT_SECURE_NO_DEPRECATE  -I `cygpath -u "$HPC_SDK\Inc"` -I `cygpath -u "$HPC_SDK\Include"`"
CMK_LD_FLAGS="$CMK_CC_FLAGS -L `cygpath -u "$HPC_SDK\Lib\amd64"` -lmsmpi"
CMK_LDXX_FLAGS="$CMK_CXX_FLAGS -L `cygpath -u "$HPC_SDK\Lib\amd64"` -lmsmpi"

if test "$NO_WIN_HPC_HEADERS_FOR_AMPI" = "1" ; then
	CMK_CC="$CHARMBIN/unix2nt_cc"
	CMK_CXX="$CHARMBIN/unix2nt_cc"
	CMK_CC_FLAGS="-D_CRT_SECURE_NO_DEPRECATE"
	CMK_CXX_FLAGS="-D_CRT_SECURE_NO_DEPRECATE"
fi

CMK_SEQ_CC="$CMK_CC $CMK_CC_FLAGS"
CMK_SEQ_CXX="$CMK_CXX $CMK_CXX_FLAGS"
CMK_SEQ_LD="$CMK_LD $CMK_LD_FLAGS"
CMK_SEQ_LDXX="$CMK_LDXX $CMK_LDXX_FLAGS"

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
