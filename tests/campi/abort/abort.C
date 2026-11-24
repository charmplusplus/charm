// campi test: MPI_Abort ends the program with its error code as the exit
// status. In campi the aborting rank runs the scheduler from inside its
// coroutine until the process exits.
// Usage: abort <rank|all> <code>
// Rank <rank> calls MPI_Abort(MPI_COMM_WORLD, <code>) after a barrier while
// every other rank waits in MPI_Recv for a message that is never sent; with
// "all", every rank calls MPI_Abort after the barrier. No rank may continue.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mpi.h"
#include "campi.h"

CAMPI_MAIN(argc, argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  bool all = (argc > 1) && strcmp(argv[1], "all") == 0;
  int abortRank = (argc > 1 && !all) ? atoi(argv[1]) : 0;
  int code = (argc > 2) ? atoi(argv[2]) : 1;
  if (abortRank < 0 || abortRank >= size) {
    if (rank == 0) printf("campi abort: rank %d out of range (%d ranks)\n", abortRank, size);
    MPI_Finalize();
    co_return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (all || rank == abortRank) {
    MPI_Abort(MPI_COMM_WORLD, code);
    printf("campi abort: MPI_Abort returned\n");
    co_return 1;
  }

  int buf;
  MPI_Recv(&buf, 1, MPI_INT, abortRank, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printf("campi abort: rank %d continued after MPI_Abort\n", rank);
  co_return 1;
}
