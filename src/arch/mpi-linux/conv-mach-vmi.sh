#in vmi, no mpicc, so swap mpicc to gcc if no overriding.
override=0
vmidir=gcc
case "$CMK_CC" in
mpicc*) override=1 ;;
ecc*)   vmidir=ecc ;;
icc*)   vmidir=icc ;;
pgcc*)  vmidir=pgi ;;
esac

#default vmi dir
test -z "$CMK_INCDIR" && CMK_INCDIR='-I /usr/local/vmi/mpich/include'
test -z "$CMK_LIBDIR" && CMK_LIBDIR="-L /usr/local/vmi/mpich/lib/$vmidir"

if test $override = 1
then
CMK_CPP_C="gcc -E "
CMK_CC="gcc "
CMK_CC_RELIABLE="gcc "
CMK_CC_FASTEST="gcc "
CMK_CXX="g++ "
CMK_CXXPP="gcc -E "
CMK_LD="gcc "
CMK_LDXX="g++ "
fi

#append CMK_INCDIR and CMK_LIBDIR for vmi
CMK_CPP_C="$CMK_CPP_C -E  $CMK_INCDIR "
CMK_CC="$CMK_CC $CMK_INCDIR "
CMK_CC_RELIABLE="$CMK_CC_RELIABLE $CMK_INCDIR "
CMK_CC_FASTEST="$CMK_CC_FASTEST $CMK_INCDIR "
CMK_CXX="$CMK_CXX $CMK_INCDIR "
CMK_CXXPP="$CMK_CXXPP -E $CMK_INCDIR "
CMK_LD="$CMK_LD $CMK_LIBDIR "
CMK_LDXX="$CMK_LDXX $CMK_LIBDIR "

CMK_LIBS="$CMK_LIBS -lmpich -lvmi -ldl -lpthread"
