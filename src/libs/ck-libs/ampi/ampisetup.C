#include "charm-api.h"

// Default mpi_setup
// this routine is executed when no TCharmUserSetup is present.

CDECL void MPI_Setup(void);
FDECL void FTN_NAME(MPI_SETUP,mpi_setup)(void);

extern "C" void
MPI_Setup_Switch(void)
{
#if MPI_FORTRAN
	FTN_NAME(MPI_SETUP,mpi_setup)();
#else	
	MPI_Setup();
#endif	
}
