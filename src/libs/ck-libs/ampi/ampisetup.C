#include "charm-api.h"

// Default ampi_setup
// this routine is executed when no TCharmUserSetup is present.

CDECL void AMPI_Setup(void);
FDECL void FTN_NAME(AMPI_SETUP,ampi_setup)(void);

extern "C" void
AMPI_Setup_Switch(void)
{
#if AMPI_FORTRAN
	FTN_NAME(AMPI_SETUP,ampi_setup)();
#else	
	AMPI_Setup();
#endif	
}
