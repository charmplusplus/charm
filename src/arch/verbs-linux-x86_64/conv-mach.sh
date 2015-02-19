#CMK_DEFS="$CMK_DEFS -DHAVE_USR_INCLUDE_MALLOC_H=1 "
CMK_AMD64="-m64"

CMK_PIC="-fPIC"
CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="gcc -E $CMK_AMD64 "
CMK_CC="gcc $CMK_AMD64 "
CMK_CXX="g++ $CMK_AMD64 "
CMK_CXXPP="$CMK_CXX -x c++ -E $CMK_AMD64 "
CMK_XIOPTS=""
CMK_LD="$CMK_CC $CMK_AMD64 "
CMK_LDXX="$CMK_CXX $CMK_AMD64 "
CMK_LD_SHARED="-shared"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_LIBS="-lckqt"
CMK_RANLIB="ranlib"
CMK_WARNINGS_ARE_ERRORS="-Werror"
CMK_CXX_OPTIMIZE="-O3"

# native compiler for compiling charmxi, etc
CMK_NATIVE_CC="$CMK_CC $CMK_AMD64 -fPIC $CMK_DEFS "
CMK_NATIVE_CXX="$CMK_CXX $CMK_AMD64 -fPIC $CMK_DEFS "
CMK_NATIVE_LD="$CMK_CC $CMK_AMD64 -fPIC "
CMK_NATIVE_LDXX="$CMK_CXX $CMK_AMD64 -fPIC "

# native compiler for compiling charmxi, etc
CMK_SEQ_CC="$CMK_NATIVE_CC"
CMK_SEQ_CXX="$CMK_NATIVE_CXX"
CMK_SEQ_LD="$CMK_NATIVE_LD"
CMK_SEQ_LDXX="$CMK_NATIVE_LDXX"

CMK_QT='generic64-light'

CMK_CF90=`which f95 2>/dev/null`
if test -n "$CMK_CF90"
then
#    CMK_FPP="/lib/cpp -P -CC"
#    CMK_CF90="$CMK_CF90 -fpic -fautomatic -fdollar-ok "
#    CMK_CF90_FIXED="$CMK_CF90 -ffixed-form "
#    CMK_F90LIBS="-lgfortran "
#    CMK_F90_USE_MODDIR=1
#    CMK_F90_MODINC="-I"
#    CMK_MOD_NAME_ALLCAPS=
#    CMK_MOD_EXT="mod"
    . $CHARMINC/conv-mach-gfortran.sh
else
    CMK_CF77="g77 "
    CMK_CF90="f90 "
    CMK_CF90_FIXED="$CMK_CF90 -W132 "
    CMK_F90LIBS="-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math "
    CMK_F77LIBS="-lg2c "
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC="-p"
fi

#default ibverbs path for openib
if test -z "$CMK_INCDIR"
then
  #openib-1.1
  if test -f /opt/ofed/include/infiniband/verbs.h
  then
    CMK_INCDIR="-I/opt/ofed/include/"
    CMK_LIBDIR="-L/opt/ofed/lib64"
	fi
  if test -f /usr/local/ofed/include/infiniband/verbs.h
  then
    CMK_INCDIR="-I/usr/local/ofed/include/"
    CMK_LIBDIR="-L/usr/local/ofed/lib64"
	fi
fi

CMK_LIBS="$CMK_LIBS -libverbs"

