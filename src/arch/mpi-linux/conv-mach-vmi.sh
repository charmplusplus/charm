#in vmi, no mpicc, so switch to gcc if no overriding.
if test "$CMK_CC" = 'mpicc '
then
test -z "$CMK_INCDIR" && CMK_INCDIR='-I /usr/local/vmi/mpich/include'
test -z "$CMK_LIBDIR" && CMK_LIBDIR='-L /usr/local/vmi/mpich/lib/gcc'
CMK_CPP_C="gcc -E $CMK_INCDIR "
CMK_CC="gcc $CMK_INCDIR "
CMK_CC_RELIABLE="gcc $CMK_INCDIR "
CMK_CC_FASTEST="gcc $CMK_INCDIR "
CMK_CXX="g++ $CMK_INCDIR "
CMK_CXXPP="gcc -E $CMK_INCDIR "
CMK_LD="gcc $CMK_LIBDIR "
CMK_LDXX="g++ $CMK_LIBDIR "
fi
CMK_LIBS="$CMK_LIBS -lmpich -lvmi -ldl -lpthread"
