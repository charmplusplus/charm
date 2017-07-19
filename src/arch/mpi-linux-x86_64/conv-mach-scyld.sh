
case "$CMK_CC" in
mpicc*)
	CMK_CPP_C="gcc "
	CMK_CC="gcc"
	CMK_CC_RELIABLE="gcc  -fPIC"
	CMK_CC_FASTEST="gcc  -fPIC"
	CMK_CXX="g++"
	CMK_LD="gcc"
	CMK_LDXX="g++"

	CMK_CPP_C_FLAGS="-E"
	CMK_CC_FLAGS="-fPIC"
	CMK_CXX_FLAGS="-fPIC"
	CMK_LD_FLAGS=""
	CMK_LDXX_FLAGS=""

# native compiler for compiling charmxi, etc
	CMK_NATIVE_CC="$CMK_CC $CMK_CC_FLAGS"
	CMK_NATIVE_CXX="$CMK_CXX $CMK_CXX_FLAGS"
	CMK_NATIVE_LD="$CMK_CC $CMK_CC_FLAGS"
	CMK_NATIVE_LDXX="$CMK_CXX $CMK_CXX_FLAGS"

# native compiler for compiling charmxi, etc
	CMK_SEQ_CC="$CMK_CC $CMK_CC_FLAGS"
	CMK_SEQ_CXX="$CMK_CXX $CMK_CXX_FLAGS"
	CMK_SEQ_LD="$CMK_CC $CMK_CC_FLAGS"
	CMK_SEQ_LDXX="$CMK_CXX $CMK_CXX_FLAGS"
	;;
esac
CMK_LIBS="$CMK_LIBS -lmpi"
