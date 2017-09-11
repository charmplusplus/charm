ICC_STATIC=true

ICCOPTS="-static-intel"

CMK_CPP_C='icc -E '
CMK_CC='icc '
CMK_CXX='icpc '
CMK_LD="icc $ICCOPTS"
CMK_LDXX="icpc $ICCOPTS"
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"

. $CHARMINC/conv-mach-ifort.sh

CMK_COMPILER='icc'
