# Assumes Clang C/C++ compiler:
CMK_CPP_CHARM="cpp -P"
CMK_CPP_C="clang$CMK_COMPILER_SUFFIX"
CMK_CC="clang$CMK_COMPILER_SUFFIX"
CMK_LD="clang$CMK_COMPILER_SUFFIX"
CMK_CXX="clang++$CMK_COMPILER_SUFFIX"
CMK_LDXX="clang++$CMK_COMPILER_SUFFIX"

CMK_CPP_C_FLAGS="-E"

if [ "$CMK_COMPILER" = "msvc" ]; then
  CMK_AR='ar q'
  CMK_LIBS='-lws2_32 -lpsapi -lkernel32'
  CMK_SEQ_LIBS="$CMK_LIBS"

  CMK_NATIVE_CC="$CMK_CC"
  CMK_NATIVE_LD="$CMK_LD"
  CMK_NATIVE_CXX="$CMK_CXX"
  CMK_NATIVE_LDXX="$CMK_LDXX"
fi

CMK_PIC='' # empty string: will be reset to default by conv-config.sh
CMK_PIE='-fPIE'

CMK_WARNINGS_ARE_ERRORS="-Werror"

CMK_COMPILER='clang'
