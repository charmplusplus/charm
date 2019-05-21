#include <stdio.h>
#include "charm++.h"
#include "mpi.h"

void exm_mpi_fn(void* in, void* out) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[%d] In AMPI: Hello[%d] from AMPI rank %d\n", CkMyPe(), *((int*)in), rank);
}
