#in vmi, no mpicc, so switch to gcc if no overriding.
if test "$CMK_CC" = 'mpicc '
then
CMK_CPP_C='gcc -E -I/usr/local/vmi/mpich/include '
CMK_CC='gcc -I/usr/local/vmi/mpich/include '
CMK_CC_RELIABLE='gcc -I/usr/local/vmi/mpich/include '
CMK_CC_FASTEST='gcc -I/usr/local/vmi/mpich/include '
CMK_CXX='g++ -I/usr/local/vmi/mpich/include '
CMK_CXXPP='gcc -E -I/usr/local/vmi/mpich/include '
CMK_LD='gcc -L /usr/local/vmi/mpich/lib/gcc'
CMK_LDXX='g++ -L /usr/local/vmi/mpich/lib/gcc'
fi
CMK_LIBS="$CMK_LIBS -lmpich -lvmi -ldl -lpthread"
