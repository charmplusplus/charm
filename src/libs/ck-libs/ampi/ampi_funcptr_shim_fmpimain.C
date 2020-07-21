#include "charm-api.h"

CLINKAGE void FTN_NAME(MPI_MAIN,mpi_main)(void);

/*
 * Provide a symbol which we can definitely export.
 * Saves users from needing to place the following in their Fortran code:
!GCC$ ATTRIBUTES DLLEXPORT :: mpi_main
!DIR$ ATTRIBUTES DLLEXPORT :: mpi_main
 */
CLINKAGE CMI_EXPORT void AMPI_Main_fortran_export(void);
CLINKAGE CMI_EXPORT void AMPI_Main_fortran_export(void)
{
  FTN_NAME(MPI_MAIN,mpi_main)();
}
