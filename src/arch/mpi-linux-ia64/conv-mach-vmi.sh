test -z "$CMK_INCDIR" && CMK_INCDIR='-I /usr/local/vmi/mpich/include'
test -z "$CMK_LIBDIR" && CMK_LIBDIR='-L /usr/local/vmi/mpich/lib/gcc'
CMK_CC="$CMK_CC $CMK_INCDIR "
CMK_CXX="$CMK_CXX $CMK_INCDIR "
CMK_LD="$CMK_LD $CMK_LIBDIR "
CMK_LDXX="$CMK_LDXX $CMK_LIBDIR "
CMK_LIBS="$CMK_LIBS -lpmpich -lvmi -ldl -lpthread"
