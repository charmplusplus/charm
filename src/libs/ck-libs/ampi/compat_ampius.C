#include "charm-api.h"
#include "ampi.h"

extern int _ampi_fallback_setup_count;

CDECL void MPI_Setup(void)
{
	_ampi_fallback_setup_count++;
}
