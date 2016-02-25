#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv)
{
  double inval, outval;
  int rank, size, expect;
  MPI_Request req;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  inval = rank+1;
  expect = (size*(size+1))/2;

  MPI_Reduce(&inval, &outval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    if (outval == expect)
      printf("MPI_Reduce test passed\n");
    else {
      printf("MPI_Reduce test failed!\n");
      MPI_Finalize();
      return 1;
    }
  }

  MPI_Ireduce(&inval, &outval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req);
  if (rank == 0) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    if (outval == expect)
      printf("MPI_Ireduce test passed\n");
    else {
      printf("MPI_Ireduce test failed!\n");
      MPI_Finalize();
      return 1;
    }
  }

  MPI_Finalize();
  return 0;
}
