#include "charm-api.h"
#include "ampi.h"

FDECL void FTN_NAME(MPI_MAIN,mpi_main)(int argc,char **argv);

FDECL void FTN_NAME(MPI_SETUP,mpi_setup)(void)
{
	MPI_Register_main(FTN_NAME(MPI_MAIN,mpi_main),"default");
}

