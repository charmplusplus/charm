# for Intel Fortran compiler 8.0 and higher which is renamed to ifort from ifc
CMK_CF77='ifort -auto '
CMK_CF90='ifort -auto '
CMK_CF90_FIXED="$CMK_CF90 -132 -FI "
CMK_F90LIBS="-L/opt/intel_fc_80/lib/ -lifcore -lifport "
CMK_MOD_NAME_ALLCAPS=
CMK_MOD_EXT="mod"
CMK_F90_USE_MODDIR=""
