BGQ_ZLIB=/soft/libraries/alcf/current/xl/ZLIB/
BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
BGQ_INC="-I$BGQ_ZLIB/include"
BGQ_LIB="-L$BGQ_ZLIB/lib -lpthread -lrt" 
CMK_SYSLIBS="$BGQ_LIB"
  
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

CMK_CC="mpixlc_r -qcpluscmt -qhalt=e $BGQ_INC -qnokeyword=__int128 -qsmp=noostls"
CMK_CXX="mpixlcxx_r -qhalt=e $BGQ_INC -qnokeyword=__int128 -qsmp=noostls"
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_CF77="mpixlf77_r "
CMK_CF90="mpixlf90_r  -qsuffix=f=f90" 
CMK_CF90_FIXED="mpixlf90_r " 
CMK_C_OPTIMIZE='-O3'
CMK_CXX_OPTIMIZE='-O3'
CMK_ENABLE_C11='-qlanglvl=extc1x'
CMK_ENABLE_CPP11='-qlanglvl=extended0x'

CMK_AR='ar cq'
CMK_NM='nm '
CMK_QT="aix"
CMK_NATIVE_LD="$CMK_NATIVE_CC"
CMK_NATIVE_LDXX="$CMK_NATIVE_CXX"
CMK_RANLIB="ranlib"
CMK_F90LIBS="-lxlf90 -lxlopt -lxl -lxlfmath"

