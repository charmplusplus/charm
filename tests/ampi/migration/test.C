/**
 * AMPI Thread Migration Test
 * Migrate the rank 1 AMPI process from node to node in order.
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "charm.h" /* For CkAbort */


int main(int argc,char **argv)
{

  int rank;            /* process id */
  int p;                /* number of processes */

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &p );

  MPI_Barrier(MPI_COMM_WORLD);

  if(p >= 1){
    CkPrintf("\nbegin migrating\n");

    for (int step=0; step<=CkNumPes(); ++step) {
      if (rank == 1) {
	int destination_pe = (CkMyPe() + 1) % CkNumPes();
	CkPrintf("Trying to migrate partition %d from pe %d to %d\n",
		 rank, CkMyPe(), destination_pe);
	//fflush(stdout);
	CkAssert(destination_pe >= 0);
	int migrate_test = CkMyPe();
	printf("Entering TCHARM_Migrate_to, "
               "FEM_My_partition is %d, "
	       "CkMyPe() is %d, migrate_test is %d\n",
	       rank, CkMyPe(), migrate_test);
	//fflush(stdout);
	AMPI_Migrate_to_pe(destination_pe);
	printf("Leaving TCHARM_Migrate_to, "
               "FEM_My_partition is %d, "
	       "CkMyPe() is %d, migrate_test is %d\n",
	       rank, CkMyPe(), migrate_test);
	//fflush(stdout);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      CkPrintf("Done with step %d\n", step);
      //fflush(stdout);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    CkPrintf("done migrating\n");
    MPI_Barrier(MPI_COMM_WORLD);

  }

  if (rank==0) CkPrintf("All tests passed\n");
  MPI_Finalize();
  return 0;
}
