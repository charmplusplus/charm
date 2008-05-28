AMDOPTS="-tp barcelona-64 "

CMK_CPP_C="pgcc -E "
CMK_CC="pgcc -fPIC $AMDOPTS "
CMK_CC_RELIABLE="gcc "
CMK_CXX="pgCC -fPIC $AMDOPTS "
CMK_CXXPP="pgCC -E $AMDOPTS "
CMK_LD="$CMK_CC $AMDOPTS "
CMK_LDXX="$CMK_CXX $AMDOPTS "

# compiler for compiling sequential programs
# pgcc can not handle QT right for generic64, so always use gcc
CMK_SEQ_CC="gcc "
CMK_SEQ_LD="gcc "
CMK_SEQ_CXX="pgCC "
CMK_SEQ_LDXX="$CMK_SEQ_CXX"
CMK_SEQ_LIBS=""

# compiler for native programs
CMK_NATIVE_CC="gcc "
CMK_NATIVE_LD="gcc "
CMK_NATIVE_CXX="g++ "
CMK_NATIVE_LDXX="g++ "
CMK_NATIVE_LIBS=""

# fortran compiler
CMK_CF77="pgf77 "
CMK_CF90="pgf90 "
CMK_CF90_FIXED="$CMK_CF90 -Mfixed "
CMK_F90LIBS="-L$PGI/linux86-64/7.1-2/lib  -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl -lpgftnrtl "
CMK_F90_USE_MODDIR=""
CPPFLAGS="$CPPFLAGS -fPIC $AMDOPTS "
