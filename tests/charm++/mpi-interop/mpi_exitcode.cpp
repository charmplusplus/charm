// MPI main() of a Charm++ program with a mainchare (exitcode.C). The whole
// program runs inside CharmLibInit, until it calls CkExit(code). CharmLibExit
// returns that code on every process, and main returns it after MPI_Finalize.
// Usage: mpi_exitcode <code> [Charm++ args].
#include <mpi.h>
#include <stdio.h>
#include "mpi-interoperate.h"

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CharmLibInit(MPI_COMM_WORLD, argc, argv);
  int code = CharmLibExit();

  // MPI still works after CharmLibExit.
  int maxCode = -1;
  MPI_Allreduce(&code, &maxCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  printf("rank %d exit code %d\n", rank, code);
  fflush(stdout);
  MPI_Finalize();
  return code;
}
