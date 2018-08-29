# NOTE: cc-clang.sh is used by default on BGQ rather than this

. $CHARMINC/cc-bluegene.sh

BGQ_INC="-I$BGQ_ZLIB/include"
BGQ_LIB="-L$BGQ_ZLIB/lib -lpthread -lrt"

usesGCC=`cat $CHARMINC/conv-mach-opt.h  | grep "cc-gcc"`;
if [[ -z $usesGCC ]]
then
  if [[ -z `command -v mpixlcxx` ]]
  then
    echo "mpixlcxx not in default path; please load xl wrappers for MPI" >&2
    exit 1
  fi

  compiler=`mpixlcxx_r -show | awk '{print $1}'`;

  if [[ $compiler == "bgxlC_r" ]] ; then
    true
  else
    echo "mpixlcxx_r does not use bgxlC_r, uses $compiler; please load xl wrappers for MPI" >&2
    exit 1
  fi
fi

GCC_OPTS="$GCC_OPTS -mminimal-toc $BGQ_INC"

CMK_CXX="mpixlcxx_r"
CMK_CC="mpixlc_r"
CMK_GCXX="mpicxx $GCC_OPTS "

CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"

CMK_CC_FLAGS="-qcpluscmt -qhalt=e $BGQ_INC -qnokeyword=__int128 -qtls=local-exec"
CMK_CXX_FLAGS="-qhalt=e $BGQ_INC -qnokeyword=__int128 -qtls=local-exec -DCMK_USING_XLC=1"
CMK_LD_FLAGS="$CMK_CC_FLAGS"
CMK_LDXX_FLAGS="$CMK_CXX_FLAGS"

CMK_CF77="mpixlf77_r "
CMK_CF90="mpixlf90_r  -qsuffix=f=f90"
CMK_CF90_FIXED="mpixlf90_r "

CMK_SYSLIBS="$BGQ_LIB"

CMK_COMPILER='bgxlc'
CMK_CCS_AVAILABLE='0'
