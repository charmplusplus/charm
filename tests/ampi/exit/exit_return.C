#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  printf("In application, expecting to exit with exit code 42.\n");

  return 42;
}
