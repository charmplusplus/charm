#include "charm-api.h"
#include "ampi.h"

CDECL void AMPI_Main(int argc,char **argv);

CDECL void AMPI_Setup(void)
{
	AMPI_Register_main(AMPI_Main,"default");
}

