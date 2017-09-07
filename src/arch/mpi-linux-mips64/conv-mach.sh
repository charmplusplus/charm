. $CHARMINC/cc-mpi.sh

CMK_INCDIR='/sicortex/software/gentoo/buildroot/default/usr/include'
CMK_LIBDIR='-L/sicortex/software/gentoo/buildroot/default/lib64 -L/sicortex/software/gentoo/buildroot/default/usr/lib64'

CMK_CPP_C_FLAGS="$CMK_CPP_C_FLAGS -G0"
CMK_CC_FLAGS="$CMK_CC_FLAGS -G0 -mips64 -march=5kf -mtune=5kf"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -G0 -mips64 -march=5kf -mtune=5kf"

CMK_LIBS="$CMK_LIBS -lscdma -lpmi "
#CMK_QT="i386-gcc"

CMK_NATIVE_CC='mpicc'
CMK_NATIVE_CXX='mpicxx'
CMK_NATIVE_LD='mpicc'
CMK_NATIVE_LDXX='mpicxx'
CMK_NATIVE_LIBS=''

CMK_NATIVE_CC_FLAGS='-G0 -mips64'
CMK_NATIVE_CXX_FLAGS='-G0 -mips64'
CMK_NATIVE_LD_FLAGS='-G0 -mips64'
CMK_NATIVE_LDXX_FLAGS='-G0 -mips64'

CMK_CF77='mpif77'
CMK_CF90='mpif90'
#CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT='mod'
