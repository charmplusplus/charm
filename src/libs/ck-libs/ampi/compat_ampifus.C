#include "charm-api.h"
#include "ampi.h"

extern int _ampi_fallback_setup_count;

FDECL void FTN_NAME(MPI_SETUP,mpi_setup)(void)
{
	_ampi_fallback_setup_count++;
}
