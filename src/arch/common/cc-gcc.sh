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
CMK_LIBS='-lckqt'
CMK_PIC='-fPIC'

if [ "$CMK_MACOSX" ]; then
  # find real gcc (not Apple's clang) in $PATH on darwin, works with homebrew/macports
  candidates=$(which gcc gcc-{4..19} gcc-mp-{4..19} 2>/dev/null)
  for cand in $candidates; do
    $cand -v 2>&1 | grep -q clang
    if [ $? -eq 1 ]; then
      cppcand=$(echo $cand | sed s,cc,++,)
      CMK_CPP_C="$cand"
      CMK_CC="$cand "
      CMK_LD="$cand "
      CMK_CXX="$cppcand "
      CMK_LDXX="$cppcand "

      CMK_CC_FLAGS="-fPIC"
      CMK_CXX_FLAGS="-fPIC -Wno-deprecated"
      CMK_LD_FLAGS="-fPIC -Wl,-no_pie "
      CMK_LDXX_FLAGS="-fPIC -multiply_defined suppress -Wl,-no_pie"
      found=1
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "No suitable non-clang gcc found, exiting"
    exit 1
  fi
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
