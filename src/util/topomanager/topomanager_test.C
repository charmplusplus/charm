#include "TopoManager.h"
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int numranks, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int i, ndims=0, *dims;

  TopoManager_init(numranks);
  TopoManager_getDimCount(&ndims);
  if (ndims <= 0) {
    printf("ERROR: Rank %d got negative number of dimensions\n", myrank);
    MPI_Finalize();
    return 0;
  }
  dims = (int*)malloc(sizeof(int)*(ndims+1));

  if (myrank == 0) {
    printf("Testing TopoManager...\n");
    printf("MPI Job Size: %d ranks\n\n", numranks);
    printf("Machine topology has %d dimensions\n", ndims);

    TopoManager_getDims(dims);
    printf("Torus Size ");
    for (i=0; i < ndims; i++) printf("[%d] ", dims[i]);
    printf("\n\n");

    FILE *out = fopen("allocationC.txt","w");
    TopoManager_printAllocation(out);
    fclose(out);
    printf("Dumped allocation to allocationC.txt\n");
  }

  TopoManager_getPeCoordinates(myrank, dims);
  printf("---- Rank %d coordinates ---> (", myrank);
  for (i=0; i < ndims-1; i++) printf("%d,", dims[i]);
  printf("%d)\n", dims[ndims-1]);
  int obtained;
  TopoManager_getPeRank(&obtained, dims);
  if (obtained != myrank)
    printf("ERROR: Failure to obtain rank from my coordinates at rank %d!!!\n", myrank);

  free(dims);
  TopoManager_free();

  MPI_Finalize();
  return 0;
}
