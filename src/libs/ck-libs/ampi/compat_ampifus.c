#include "charm-api.h"
#include "ampi.h"

FDECL void FTN_NAME(AMPI_MAIN,ampi_main)(int argc,char **argv);

FDECL void FTN_NAME(AMPI_SETUP,ampi_setup)(void)
{
	AMPI_Register_main(FTN_NAME(AMPI_MAIN,ampi_main),"default");
}

