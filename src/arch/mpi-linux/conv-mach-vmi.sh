#in vmi, no mpicc, so switch to gcc if no overriding.
if test "$CMK_CC" = 'mpicc '
then
CMK_CPP_C='gcc -E '
CMK_CC='gcc'
CMK_CC_RELIABLE='gcc '
CMK_CC_FASTEST='gcc '
CMK_CXX='g++'
CMK_CXXPP='gcc -E '
CMK_LD='gcc'
CMK_LDXX='g++'
fi
CMK_CC="$CMK_CC -I/usr/local/vmi/mpich/include "
CMK_CXX="$CMK_CXX -I/usr/local/vmi/mpich/include "
CMK_LD="$CMK_LD -L /usr/local/vmi/mpich/lib/gcc"
CMK_LDXX="$CMK_LDXX -L /usr/local/vmi/mpich/lib/gcc"
CMK_LIBS="$CMK_LIBS -lmpich -lvmi -ldl -lpthread"
