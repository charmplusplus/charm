
case "$CMK_CC" in
mpicc*)
	CMK_CPP_C="gcc -E "
	CMK_CC="gcc -fPIC"
	CMK_CC_RELIABLE="gcc  -fPIC"
	CMK_CC_FASTEST="gcc  -fPIC"
	CMK_CXX="g++ -fPIC"
	CMK_CXXPP="gcc -E "
	CMK_LD="gcc"
	CMK_LDXX="g++"

# native compiler for compiling charmxi, etc
	CMK_NATIVE_CC="$CMK_CC"
	CMK_NATIVE_CXX="$CMK_CXX"
	CMK_NATIVE_LD="$CMK_CC"
	CMK_NATIVE_LDXX="$CMK_CXX"

# native compiler for compiling charmxi, etc
	CMK_SEQ_CC="$CMK_CC"
	CMK_SEQ_CXX="$CMK_CXX"
	CMK_SEQ_LD="$CMK_CC"
	CMK_SEQ_LDXX="$CMK_CXX"
	;;
esac   
CMK_LIBS="$CMK_LIBS -lmpi"

