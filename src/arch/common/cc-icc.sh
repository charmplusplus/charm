CMK_CPP_C='icc -E '
CMK_CC='icc '
CMK_CXX='icpc '
CMK_CXXPP='icpc -E '
CMK_LD='icc -i_dynamic '
CMK_LDXX='icpc -i_dynamic '
CMK_LD_LIBRARY_PATH="-Wl,-rpath,$CHARMLIBSO/"
CMK_CF90='ifc -auto '
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
CMK_SEQ_F90="$CMK_CF90"
CMK_SEQ_CC="$CMK_CC"
CMK_SEQ_CXX="$CMK_CXX"
CMK_SEQ_LD="$CMK_LD"
CMK_SEQ_LDXX="$CMK_LDXX"
CMK_F90LIBS='-L/usr/local/intel/compiler70/ia32/lib -L/opt/intel/compiler70/ia32/lib -lintrins -lIEPCF90 -lF90 -lintrins -limf  '
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=""
