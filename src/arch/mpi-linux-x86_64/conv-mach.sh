
CMK_REAL_COMPILER=`mpiCC -show 2>/dev/null | cut -d' ' -f1 `
case "$CMK_REAL_COMPILER" in
g++) CMK_AMD64="-m64 -fPIC" ;;
esac

CMK_CPP_CHARM="/lib/cpp -P"
CMK_CPP_C="mpicc -E"
CMK_CC="mpicc $CMK_AMD64 "
CMK_CXX="mpiCC $CMK_AMD64 "
CMK_CXXPP="mpiCC -E $CMK_AMD64 "

CMK_SYSLIBS="-lmpich "
CMK_LIBS="-lckqt $CMK_SYSLIBS "
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_NATIVE_CC="gcc $CMK_AMD64 "
CMK_NATIVE_LD="gcc $CMK_AMD64 "
CMK_NATIVE_CXX="g++ $CMK_AMD64 "
CMK_NATIVE_LDXX="g++ $CMK_AMD64 "
CMK_NATIVE_LIBS=""

# fortran compiler 
CMK_CF77="f77"
CMK_CF90="f90"
CMK_F90LIBS="-L/usr/absoft/lib -L/opt/absoft/lib -lf90math -lfio -lU77 -lf77math "
CMK_F77LIBS="-lg2c "
CMK_F90_USE_MODDIR=1
CMK_F90_MODINC="-p"

CMK_QT='generic64'
CMK_RANLIB="ranlib"

