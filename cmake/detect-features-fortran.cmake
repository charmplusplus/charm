# Check Fortran naming scheme
set(CMK_FORTRAN_USES_TWOSCORE 0)
set(CMK_FORTRAN_USES_ONESCORE 0)
set(CMK_FORTRAN_USES_NOSCORE 0)
set(CMK_FORTRAN_USES_ALLCAPS 0)

if(${CMK_CAN_LINK_FORTRAN})
  if(${FortranCInterface_GLOBAL_SUFFIX} STREQUAL "__")
    set(CMK_FORTRAN_USES_TWOSCORE 1)
  elseif(${FortranCInterface_GLOBAL_SUFFIX} STREQUAL "_")
    set(CMK_FORTRAN_USES_ONESCORE 1)
  elseif(${FortranCInterface_GLOBAL_SUFFIX} STREQUAL "")
    set(CMK_FORTRAN_USES_NOSCORE 1)
  elseif(${FortranCInterface_GLOBAL_CASE} STREQUAL "UPPER")
    set(CMK_FORTRAN_USES_ALLCAPS 1)
  endif()
endif()

# Fortran module names
set(CMK_MOD_NAME_ALLCAPS ${CMK_BLUEGENEQ})
set(CMK_MOD_EXT mod)
