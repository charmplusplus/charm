// MPI program that uses Charm++ as a library: MPI phases alternate with
// Charm++ phases, then CharmLibExit returns and MPI continues until
// MPI_Finalize. Usage: mpi_interop <processes started by the launcher>
// [abort] [Charm++ args]; the test fails if MPI or Charm++ sees a different
// count. With "abort", the first Charm++ phase calls CkAbort on the last PE,
// which must end every process through MPI_Abort (Makefile: testabort).
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi-interoperate.h"
#include "phaselib.h"

static void check(bool ok, int rank, const char *what, long got, long want) {
  if (!ok) {
    printf("FAIL: rank %d: %s: got %ld, expected %ld\n", rank, what, got, want);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

int main(int argc, char **argv) {
  // SERIALIZED: with several PEs per process, an abort calls MPI_Abort from a
  // PE thread that MPI did not create (see CharmLibInit).
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (provided < MPI_THREAD_SERIALIZED && rank == 0)
    fprintf(stderr, "Warning: MPI provides thread level %d, below "
            "MPI_THREAD_SERIALIZED\n", provided);

  int expected = argc > 1 ? atoi(argv[1]) : 0;
  if (expected <= 0) {
    fprintf(stderr, "usage: %s <number of processes> [Charm++ args]\n",
            argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (size != expected) {
    fprintf(stderr,
            "FAIL: MPI_COMM_WORLD has %d ranks, expected %d: the launcher "
            "does not match the MPI library, so each process may be its own "
            "MPI job\n", size, expected);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Drop the count and the mode so Charm++ sees only its own arguments.
  bool abortMode = argc > 2 && strcmp(argv[2], "abort") == 0;
  int drop = abortMode ? 2 : 1;
  for (int i = 1; i + drop <= argc; i++) argv[i] = argv[i + drop];
  argc -= drop;

  CharmLibInit(MPI_COMM_WORLD, argc, argv);
  check(CkNumNodes() == size, rank, "Charm++ processes", CkNumNodes(), size);
  if (rank == 0)
    printf("CharmLibInit done: %d MPI ranks, %d Charm++ processes, %d PEs\n",
           size, CmiNumNodes(), CkNumPes());

  if (abortMode) {
    // For a comparison run: without CharmLibInit's hook, only the aborting
    // process ends (reconverse #321).
    if (getenv("MPI_INTEROP_NO_ABORT_HOOK")) CmiSetAbortHook(NULL);
    if (rank == 0) {
      printf("abort phase: the last PE calls CkAbort\n");
      fflush(stdout);
    }
    PhaseAbort();
    printf("FAIL: rank %d returned from the abort phase\n", rank);
    fflush(stdout);
    return 1;
  }

  for (int iter = 1; iter <= 3; iter++) {
    long mine = rank + iter, sum = 0;
    MPI_Allreduce(&mine, &sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    long want = (long)size * (size - 1) / 2 + (long)size * iter;
    check(sum == want, rank, "MPI allreduce", sum, want);

    PhaseRun(iter);

    long charmSum = PhaseResult();
    MPI_Bcast(&charmSum, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    check(charmSum == PhaseExpected(iter), rank, "Charm++ reduction", charmSum,
          PhaseExpected(iter));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
      printf("iteration %d: MPI sum %ld and Charm++ sum %ld correct\n", iter,
             sum, charmSum);
  }

  CharmLibExit();

  long one = 1, total = 0;
  MPI_Allreduce(&one, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  check(total == size, rank, "MPI allreduce after CharmLibExit", total, size);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("PASS\n");
    fflush(stdout);
  }
  MPI_Finalize();
  if (rank == 0) printf("MPI_Finalize returned\n");
  return 0;
}
