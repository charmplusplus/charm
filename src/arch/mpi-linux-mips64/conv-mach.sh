CMK_INCDIR='/sicortex/software/gentoo/buildroot/default/usr/include'
CMK_LIBDIR='-L/sicortex/software/gentoo/buildroot/default/lib64 -L/sicortex/software/gentoo/buildroot/default/usr/lib64'

CMK_CPP_CHARM='/lib/cpp -P'
CMK_CPP_C='mpicc -G0 -E '
CMK_CC='mpicc -G0 -mips64 -march=5kf -mtune=5kf '
CMK_CXX='mpicxx -G0 -mips64 -march=5kf -mtune=5kf '
CMK_CXXPP='mpicxx -G0 -E '
CMK_RANLIB='ranlib'
CMK_LIBS='-lckqt -lscdma -lpmi '
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
#CMK_QT="i386-gcc"

CMK_NATIVE_CC='mpicc -G0 -mips64 '
CMK_NATIVE_CXX='mpicxx -G0 -mips64 '
CMK_NATIVE_LD='mpicc -G0 -mips64 '
CMK_NATIVE_LDXX='mpicxx -G0 -mips64 '
CMK_NATIVE_LIBS=''

CMK_CF77='mpif77'
CMK_CF90='mpif90'
#CMK_F90LIBS='-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"
