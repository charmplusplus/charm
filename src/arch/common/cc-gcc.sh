CMK_CPP_CHARM='cpp -P'
CMK_CPP_C='gcc -E'
CMK_CC='gcc'
CMK_CXX='g++'
CMK_LD='gcc'
CMK_LDXX='g++'

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
      CMK_CPP_C="$cand -E "
      CMK_CC="$cand -fPIC "
      CMK_LD="$cand "
      CMK_CXX="$cppcand -fPIC -Wno-deprecated "
      CMK_LDXX="$cppcand "
      found=1
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "No suitable non-clang gcc found, exiting"
    exit 1
  fi
fi
