vmidir=gcc
case "$CMK_CC" in
ecc*)   vmidir=ecc6 ;;
icc*)   vmidir=icc ;;
pgcc*)  vmidir=pgi ;;
esac   

test -z "$CMK_INCDIR" && CMK_INCDIR='-I /usr/local/vmi/mpich/include'
test -z "$CMK_LIBDIR" && CMK_LIBDIR="-L /usr/local/vmi/mpich/lib/$vmidir"

CMK_CC="$CMK_CC $CMK_INCDIR "
CMK_CXX="$CMK_CXX $CMK_INCDIR "
CMK_LD="$CMK_LD $CMK_LIBDIR "
CMK_LDXX="$CMK_LDXX $CMK_LIBDIR "

CMK_LIBS="$CMK_LIBS -lvmi -ldl -lpthread"
