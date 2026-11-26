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

  // MPI still works after CharmLibExit, and every rank has the same code.
  int minCode = 0, maxCode = 0;
  MPI_Allreduce(&code, &minCode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&code, &maxCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  printf("rank %d exit code %d\n", rank, code);
  bool same = minCode == code && maxCode == code;
  if (!same)
    printf("FAIL: rank %d has exit code %d, but the codes of all ranks range "
           "from %d to %d\n", rank, code, minCode, maxCode);
  fflush(stdout);
  MPI_Finalize();
  return same ? code : 99;
}
