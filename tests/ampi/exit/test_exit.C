#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  printf("In application, expecting to exit with exit code 42.\n");
  exit(42);

  // Never reached:
  MPI_Finalize();

  printf("After application.\n");
  return 0;
}
