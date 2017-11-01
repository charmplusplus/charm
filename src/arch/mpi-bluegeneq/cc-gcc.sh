BGQ_BIN=$BGQ_FLOOR/gnu-linux/bin
BGQ_ZLIB=/soft/libraries/alcf/current/gcc/ZLIB/

BGQ_INC="-I$BGQ_ZLIB/include"
BGQ_LIB="-L$BGQ_ZLIB/lib -lpthread -lrt"

if [[ -z `command -v mpicxx` ]]
then
  echo "mpicxx not in default path; please load gcc wrappers for MPI" >&2
  exit 1
fi

compiler=`mpicxx -show | awk '{print $1}'`;
if [[ $compiler == "powerpc64-bgq-linux-g++" ]] ; then
  true
else
  echo "mpicxx does not use g++, uses $compiler; please load gcc wrappers for MPI" >&2
  exit 1
fi

GCC_OPTS="-Wno-deprecated -mminimal-toc $BGQ_INC"

CMK_CPP_CHARM="$BGQ_BIN/powerpc64-bgq-linux-cpp -P"
CMK_CPP_C="$BGQ_BIN/powerpc64-bgq-linux-cpp"
CMK_CXX='mpicxx'
CMK_CC='mpicc'
CMK_CF77="mpif77 "
CMK_CF90='mpif90'
CMK_RANLIB="$BGQ_BIN/powerpc64-bgq-linux-ranlib "
CMK_AR="$BGQ_BIN/powerpc64-bgq-linux-ar q "
CMK_SYSLIBS="$BGQ_LIB"
CMK_LIBS='-lckqt'
CMK_LD="$CMK_CC"
CMK_LDXX="$CMK_CXX"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC='gcc'
CMK_NATIVE_LD='gcc'
CMK_NATIVE_CXX='g++'
CMK_NATIVE_LDXX='g++'
CMK_F90LIBS='-lgfortran '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_QT="generic64-light"

CMK_CPP_C_FLAGS='-E'

CMK_CC_FLAGS="$GCC_OPTS"
CMK_CXX_FLAGS="$GCC_OPTS"
CMK_LD_FLAGS=''
CMK_LDXX_FLAGS=''

CMK_NATIVE_CC_FLAGS='-Wno-deprecated'
CMK_NATIVE_CXX_FLAGS='-Wno-deprecated'
CMK_NATIVE_LD_FLAGS=''
CMK_NATIVE_LDXX_FLAGS=''

CMK_COMPILER='gcc'
