#include "charm-api.h"
#include "ampi.h"

CDECL void MPI_Main(int argc,char **argv);

CDECL void MPI_Setup(void)
{
	MPI_Register_main(MPI_Main,"default");
}

