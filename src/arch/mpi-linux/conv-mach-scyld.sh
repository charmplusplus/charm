
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
	;;
esac
CMK_LIBS="$CMK_LIBS -lmpi"
