#in vmi, no mpicc, so switch to gcc if no overriding.
override=0
case "$CMK_CC" in
mpicc*) override=1 ;;
esac
if test $override = 1
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
