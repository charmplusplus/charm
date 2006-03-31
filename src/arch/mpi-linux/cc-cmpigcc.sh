CMK_CPP_C="cmpicc -E"
CMK_CC="cmpicc -gm -ccl gcc "
CMK_CXX="cmpicc -gm -ccl g++ "
CMK_CXXPP="cmpicc -E -ccl g++ "
CMK_LD="cmpicc -ccl gcc -gm "
CMK_LDXX="cmpicc -ccl g++ -gm "
CMK_LIBS="-lckqt -lcmpi "
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

CMK_CF77="cmpifc -gnu "
CMK_CF90="cmpifc -gnu "
CMK_F90LIBS='-lifcore  '
CMK_F90_USE_MODDIR=
CMK_F90_MODINC=""
