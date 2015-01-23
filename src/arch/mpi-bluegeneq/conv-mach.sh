BGQ_TYPICAL_FLOOR=/bgsys/drivers/ppcfloor

# if no floor set, use typical floor path
if test -z "$BGQ_FLOOR"
then
  BGQ_FLOOR=$BGQ_TYPICAL_FLOOR
fi

# if no install path (for experimental) set, use floor
if test -z "$BGQ_INSTALL"
then
  BGQ_INSTALL=$BGQ_TYPICAL_FLOOR
fi

BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
BGQ_ZLIB=/soft/libraries/alcf/current/xl/ZLIB/

BGQ_INC="-I$BGQ_ZLIB/include"
BGQ_LIB="-L$BGQ_ZLIB/lib -lpthread -lrt" 

CMK_SYSLIBS="$BGQ_LIB"

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

OPTS_CPP="$OPTS_CPP"
GCC_OPTS="-Wno-deprecated -mminimal-toc $BGQ_INC"
OPTS_LD="$OPTS_LD"

CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="$BGQ_BIN/powerpc64-bgq-linux-cpp -E "
CMK_CXX="mpixlcxx_r -qhalt=e $BGQ_INC -qnokeyword=__int128 -qtls=local-exec"
CMK_GCXX="mpicxx $GCC_OPTS "
CMK_CC="mpixlc_r -qcpluscmt -qhalt=e $BGQ_INC -qnokeyword=__int128 -qtls=local-exec"
CMK_CXXPP="mpicxx -E "
CMK_CF77="mpixlf77_r "
CMK_CF90="mpixlf90_r  -qsuffix=f=f90" 
CMK_CF90_FIXED="mpixlf90_r " 
CMK_RANLIB="$BGQ_BIN/powerpc64-bgq-linux-ranlib "
CMK_AR="$BGQ_BIN/powerpc64-bgq-linux-ar q "
CMK_SYSLIBS="$BGQ_LIB"
CMK_LIBS='-lckqt'
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'
CMK_ENABLE_C11='-qlanglvl=extc1x'
CMK_ENABLE_CPP11='-qlanglvl=extended0x'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC='gcc '
CMK_NATIVE_LD='gcc '
CMK_NATIVE_CXX='g++ -Wno-deprecated '
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS="-lxlf90 -lxlopt -lxl -lxlfmath"
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_QT="aix"
CMK_NM='nm '
