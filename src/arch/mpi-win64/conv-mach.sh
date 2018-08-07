
CMK_CPP_CHARM="cpp -P"
CMK_CPP_C_FLAGS="$CMK_CC_FLAGS -E"

CMK_CF77="f77"
CMK_CF90="f90"

CMK_XIOPTS=""
CMK_F90LIBS="-lvast90 -lg2c"

CMK_POST_EXE=".exe"
CMK_QT="none"

. $CHARMINC/cc-msvc.sh

MSMPI_SUFFIX_LIB='\amd64'
MSMPI_SUFFIX_INC=''

if test -n "$CCP_LIB64"
then
  HPC_SDK="$CCP_LIB64\..\.."
elif test -d "c:\Program Files (x86)\Microsoft SDKs\MPI"
then
  HPC_SDK="c:\Program Files (x86)\Microsoft SDKs\MPI"
  MSMPI_SUFFIX_LIB="\x64"
  MSMPI_SUFFIX_INC="\x64"
else
  HPC_SDK="c:\Program Files\Microsoft MPI"
fi
HPC_SDK=`cygpath -d "$HPC_SDK"`

# These include paths for MS MPI (added through the $INCLUDE variable) have a
# lower priority than paths added via -I, thus allowing us to use AMPI's mpi.h
# when compiling AMPI applications.
export INCLUDE="$INCLUDE;`cygpath -wl "$HPC_SDK\Inc"`;`cygpath -wl "$HPC_SDK\Include"`"
MSMPI_INCLUDE="-I `cygpath -u "$HPC_SDK\Include"`"
if test -n "MSMPI_SUFFIX_INC"
then
  export INCLUDE="$INCLUDE;`cygpath -wl "$HPC_SDK\Include$MSMPI_SUFFIX_INC"`"
  MSMPI_INCLUDE="-I `cygpath -u "$HPC_SDK\Include$MSMPI_SUFFIX_INC"` $MSMPI_INCLUDE"
fi

CMK_SYSINC="$MSMPI_INCLUDE"

CMK_MPI_LIB="-L `cygpath -u "$HPC_SDK\Lib$MSMPI_SUFFIX_LIB"` -lmsmpi"
