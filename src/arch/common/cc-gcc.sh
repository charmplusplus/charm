CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="gcc$CMK_COMPILER_SUFFIX"
CMK_CC="gcc$CMK_COMPILER_SUFFIX"
CMK_CXX="g++$CMK_COMPILER_SUFFIX"
CMK_LD="gcc$CMK_COMPILER_SUFFIX"
CMK_LDXX="g++$CMK_COMPILER_SUFFIX"

CMK_CPP_C_FLAGS="-E"

CMK_LD_SHARED='-shared'
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_RANLIB='ranlib'
CMK_LIBS="$CMK_LIBS -lckqt"
CMK_PIC='-fPIC'
CMK_PIE='' # empty string: will be reset to default by conv-config.sh

CMK_WARNINGS_ARE_ERRORS="-Werror"

if [ "$CMK_MACOSX" ]; then
  if [ -z "$CMK_COMPILER_SUFFIX" ]; then
    # find real gcc (not Apple's clang) in $PATH on darwin, works with homebrew/macports
    candidates=$(command -v gcc gcc-{19..4} gcc-mp-{19..4} 2>/dev/null)
    for cand in $candidates; do
      $cand -v 2>&1 | grep -q clang
      if [ $? -eq 1 ]; then
        cppcand=$(echo $cand | sed s,cc,++,)
        CMK_CPP_C="$cand"
        CMK_CC="$cand "
        CMK_LD="$cand "
        CMK_CXX="$cppcand "
        CMK_LDXX="$cppcand "
        CMK_COMPILER_SUFFIX="${cand#gcc}"

        found=1
        break
      fi
    done
    if [ -z "$found" ]; then
      echo "No suitable non-clang gcc found, exiting"
      exit 1
    fi
  fi

  # keep in sync with mpi-darwin-x86_64/conv-mach.sh
  CMK_CC_FLAGS="-fPIC"
  CMK_CXX_FLAGS="-fPIC -Wno-deprecated"
  CMK_LD_FLAGS="-fPIC"
  CMK_LDXX_FLAGS="-fPIC -multiply_defined suppress"
fi

if [ "$CMK_COMPILER" = "msvc" ]; then
  CMK_AR='ar q'
  CMK_LIBS='-lws2_32 -lpsapi -lkernel32'
  CMK_SEQ_LIBS="$CMK_LIBS"

  CMK_NATIVE_CC="$CMK_CC"
  CMK_NATIVE_LD="$CMK_LD"
  CMK_NATIVE_CXX="$CMK_CXX"
  CMK_NATIVE_LDXX="$CMK_LDXX"
fi

CMK_COMPILER='gcc'

if command -v "gfortran$CMK_COMPILER_SUFFIX" 'gfortran' 'f95' >/dev/null 2>&1
then
  . $CHARMINC/conv-mach-gfortran.sh
else
  if command -v 'g77' >/dev/null 2>&1
  then
    CMK_CF77='g77'
  elif command -v 'f77' >/dev/null 2>&1
  then
    CMK_CF77='f77'
  fi
  CMK_F77LIBS='-lg2c'

  if command -v 'xlf90_r' >/dev/null 2>&1
  then
    # xlf
    CMK_CF90='xlf90_r'
    bindir=`dirname $(command -v $CMK_CF90)`
    libdir="$bindir/../lib"
    CMK_CF90="$CMK_CF90 -qpic=large -qthreaded -qlanglvl=90std -qwarn64 -qspill=32648 -qsuppress=1513-029:1518-012:1518-059 -qsuffix=f=f90:cpp=F90 "
    CMK_CF90_FIXED="$CMK_CF90 -qsuffix=f=f:cpp=F -qfixed=132 "
    CMK_F90LIBS="-L/opt/ibmcmp/xlf/11.1/bin/../../../xlsmp/1.7/lib -L$libdir -lxl -lxlf90 -lxlfmath -lxlopt -lxlsmp"
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC='-I'
  elif command -v 'f90' >/dev/null 2>&1
  then
    # absoft
    CMK_CF90='f90'
    CMK_CF90_FIXED="$CMK_CF90 -W132"
    CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math'
    CMK_F90_USE_MODDIR=1
    CMK_F90_MODINC='-p'
  fi
fi
