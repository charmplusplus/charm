
# test version
ICC_ver=`icc --version | awk -F'.' '{print $1; exit}' | awk '{print $NF}'`
test -z "$ICC_ver" && echo "ICC compiler not found!" && exit 1
#echo version:$ICC_ver

#test intel x86, AMD or IA-64
CPU_model=""
model=`cat /proc/cpuinfo | grep 'model name' | tail -1`
if echo $model | grep 'AMD' > /dev/null 2>/dev/null
then
  CPU_model="AMD"
elif echo $model | grep 'AMD' > /dev/null 2>/dev/null
then
  CPU_model="Intel"
fi
if test x$CPU_model = x
then
  model=`cat /proc/cpuinfo | grep 'arch'`
  if echo $model | grep 'IA-64' > /dev/null 2>/dev/null
  then
    CPU_model="IA-64"
  fi
fi
#echo CPU: $CPU_model

if test $ICC_ver  -ge 9
then

# for version 10 
if test x$CPU_model = "xAMD"
then
  ICCOPTS="-xO"
elif test x$CPU_model = "xIntel"
then
  ICCOPTS="-xT"
fi

CMK_CPP_C='icc -E '
CMK_CC="icc -fpic $ICCOPTS "
CMK_CXX="icpc -fpic $ICCOPTS "
CMK_CXXPP="icpc -E $ICCOPTS "

CMK_LD="icc -shared-intel $ICCOPTS "
CMK_LDXX="icpc -shared-intel $ICCOPTS "

CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_NATIVE_CC="$CMK_CC"
CMK_NATIVE_CXX="$CMK_CXX"
CMK_NATIVE_LD="$CMK_LD"
CMK_NATIVE_LDXX="$CMK_LDXX"
CPPFLAGS="$CPPFLAGS -fpic $ICCOPTS "
LDFLAGS="$LDFLAGS -shared-intel $ICCOPTS "

# for Intel Fortran compiler 8.0 and higher which is renamed to ifort from ifc
# does not work for ifc 7.0
CMK_CF77="ifort -auto -fPIC "
CMK_CF90="ifort -auto -fPIC "
#CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
#FOR 64 bit machine
CMK_CF90_FIXED="$CMK_CF90 -164 -FI "
F90DIR=`which ifort 2> /dev/null`
if test -h "$F90DIR"
then
  F90DIR=`readlink $F90DIR`
fi
if test -x "$F90DIR" 
then
  F90LIBDIR="`dirname $F90DIR`/../lib"
  F90MAIN="$F90LIBDIR/for_main.o"
fi
# for_main.o is important for main() in f90 code
CMK_F90MAINLIBS="$F90MAIN "
CMK_F90LIBS="-L$F90LIBDIR -lifcore -lifport "
CMK_F77LIBS="$CMK_F90LIBS"

CMK_F90_USE_MODDIR=""


# native compiler for compiling charmxi, etc
CMK_SEQ_CC="$CMK_NATIVE_CC"
CMK_SEQ_CXX="$CMK_NATIVE_CXX"
CMK_SEQ_LD="$CMK_NATIVE_LD"
CMK_SEQ_LDXX="$CMK_NATIVE_LDXX"

else      # version other than 10

. $CHARMINC/cc-icc8.sh

fi

CMK_C_OPENMP="-openmp"
