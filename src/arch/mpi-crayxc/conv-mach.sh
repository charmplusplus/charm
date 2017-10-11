#  Cray XC build.  October 2016.
#
#  BUILD CHARM++ on the CRAY XC:
#       COMMON FOR ALL COMPILERS
#  ======================================================
#  #  Do not set this env variable when building charm++. This will cause charm++ to fail.
#  #     setenv CRAYPE_LINK_TYPE dynamic
#
#  #  Use 8M hugepages unless explicitly building with charm++'s regularpages option.
#  module load craype-hugepages8M
#
#  #  Then load a compiler's PrgEnv module.
#  #  Do not add 'craycc', 'icc', or 'gcc' to your build options or this will fail to build.
#
#  CRAY COMPILER (CCE) BUILD
#  ========================================
#  module load PrgEnv-cray         # typically default
#  module swap cce cce/8.5.4       # cce/8.5.4 or later required
#
#  INTEL BUILD
#  ================================
#  module swap PrgEnv-cray PrgEnv-intel
#
#  GCC BUILD
#  ================================
#  module swap PrgEnv-cray PrgEnv-gnu
#
#  # Build command is the same regardless of compiler environment:
#
#  # MPI build
#  ./build charm++ mpi-crayxc smp --with-production

MPI_CRAYXC=1
CMK_BUILD_CRAY=1
PMI_LIBS="$CRAY_PMI_POST_LINK_OPTS"

PGCC=`CC -V 2>&1 | grep pgCC`
ICPC=`CC -V 2>&1 | grep Intel`
GNU=`CC -V 2>&1 | grep 'g++'`
CCE=`CC -V 2>&1 | grep 'Cray'`

CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="cc -E"
CMK_CC="cc "
CMK_CXX="CC "
CMK_LD="$CMK_CC "
CMK_LDXX="$CMK_CXX "
CMK_LIBS='-lckqt'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/,$PMI_LIBS "

# compiler for compiling sequential programs
if test -n "$PGCC"
then
CMK_CC="$CMK_CC -DCMK_FIND_FIRST_OF_PREDICATE=1 "
CMK_CXX="$CMK_CXX -DCMK_FIND_FIRST_OF_PREDICATE=1 --no_using_std "
# gcc is needed for building QT
CMK_SEQ_CC="gcc -fPIC "
CMK_SEQ_CXX="pgCC -fPIC --no_using_std "
elif test -n "$CCE"
then
CMK_CXX_OPTIMIZE=" -hipa4"   # For improved C++ performance
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="g++ -fPIC "
elif test -n "$ICPC"
then
CMK_SEQ_CC="cc -fPIC "
CMK_SEQ_CXX="CC -fPIC "
else   # gcc
CMK_SEQ_CC="gcc -fPIC"
CMK_SEQ_CXX="g++ -fPIC "
fi
CMK_SEQ_LD="$CMK_SEQ_CC "
CMK_SEQ_LDXX="$CMK_SEQ_CXX "
CMK_SEQ_LIBS=""

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

CMK_RANLIB="ranlib"
CMK_QT="generic64-light"

# for F90 compiler
if test -n "$ICPC"
then
CMK_CF77="ftn -auto "
CMK_CF90="ftn -auto "
CMK_F90LIBS="-lifcore -lifport -lifcore "
else
CMK_CF77="ftn "
CMK_CF90="ftn "
CMK_F90LIBS=""
fi

if test -n "$GNU"
then
    CMK_CF77="$CMK_CF77 -ffree-line-length-none"
    CMK_CF90="$CMK_CF90 -ffree-line-length-none"
    CMK_F90LIBS=""
fi
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-I"
CMK_MOD_EXT="mod"
