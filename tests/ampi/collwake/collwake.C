/* Thread wake-up rules for non-blocking collectives (ampi::irednResult).

   Part A: each rank issues MPI_Iallreduce (or MPI_Igather) on one
   communicator and, while that is still outstanding, a blocking
   MPI_Allreduce on another. The non-blocking collective's completion must
   not resume a thread parked in the blocking one: if it does, the blocking
   call returns before its result is written (or the later result finds no
   blocking request and crashes).

   Part B: MPI_Waitall over outstanding MPI_Iallreduce and MPI_Igather
   requests on different communicators. Waitall blocks with resumeOnRecv set, so the completion of the
   last request must resume the thread; otherwise the job hangs.

   Every result is an exact sum of rank numbers, checked against a value
   computed locally. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static int failures = 0;

static void check(int rank, int iter, const char *what, long got, long want)
{
  if (got != want) {
    printf("FAIL: rank %d iter %d: %s = %ld, expected %ld\n", rank, iter, what, got, want);
    fflush(stdout);
    failures++;
  }
}

/* Sum of the world ranks first, first+stride, ... (count terms). */
static long sumOver(int first, int stride, int count)
{
  long s = 0;
  for (int i = 0; i < count; i++) s += first + i * stride;
  return s;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int iters = (argc > 1) ? atoi(argv[1]) : 50;
  /* Bit mask of the parts to run: 1 = A1, 2 = A2, 4 = A3, 8 = B. */
  const int parts = (argc > 2) ? atoi(argv[2]) : 15;

  /* pair: ranks {0,1}, {2,3}, ...; parity: even ranks / odd ranks. */
  MPI_Comm pair, parity;
  MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &pair);
  MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &parity);
  int pairRank, pairSize, parityRank, paritySize;
  MPI_Comm_rank(pair, &pairRank);
  MPI_Comm_size(pair, &pairSize);
  MPI_Comm_rank(parity, &parityRank);
  MPI_Comm_size(parity, &paritySize);

  const long mine = rank;
  const long wantWorld = sumOver(0, 1, size);
  const long wantPair = sumOver((rank / 2) * 2, 1, pairSize);
  const long wantParity = sumOver(rank % 2, 2, paritySize);
  std::vector<long> gathered(paritySize);

  for (int iter = 0; iter < iters; iter++) {
    /* A1: non-blocking on a subcommunicator, blocking on MPI_COMM_WORLD. */
    if (parts & 1) {
      long gotPair = -1, gotWorld = -1;
      MPI_Request req;
      MPI_Iallreduce(&mine, &gotPair, 1, MPI_LONG, MPI_SUM, pair, &req);
      MPI_Allreduce(&mine, &gotWorld, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      check(rank, iter, "A1 blocking Allreduce(world)", gotWorld, wantWorld);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      check(rank, iter, "A1 Iallreduce(pair)", gotPair, wantPair);
    }

    /* A2: non-blocking on MPI_COMM_WORLD, blocking on a subcommunicator. */
    if (parts & 2) {
      long gotWorld = -1, gotParity = -1;
      MPI_Request req;
      MPI_Iallreduce(&mine, &gotWorld, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &req);
      MPI_Allreduce(&mine, &gotParity, 1, MPI_LONG, MPI_SUM, parity);
      check(rank, iter, "A2 blocking Allreduce(parity)", gotParity, wantParity);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      check(rank, iter, "A2 Iallreduce(world)", gotWorld, wantWorld);
    }

    /* A3: non-blocking Igather on a subcommunicator, blocking on world. */
    if (parts & 4) {
      long gotWorld = -1;
      for (int i = 0; i < paritySize; i++) gathered[i] = -1;
      MPI_Request req;
      MPI_Igather(&mine, 1, MPI_LONG, gathered.data(), 1, MPI_LONG, 0, parity, &req);
      MPI_Allreduce(&mine, &gotWorld, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      check(rank, iter, "A3 blocking Allreduce(world)", gotWorld, wantWorld);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (parityRank == 0)
        for (int i = 0; i < paritySize; i++)
          check(rank, iter, "A3 Igather(parity)[i]", gathered[i], rank % 2 + 2 * i);
    }

    /* B: MPI_Waitall over several non-blocking collectives, at most one per
       communicator. (Two outstanding on the same communicator, one of them an
       Igather, is a different problem: irednResult matches results to
       requests in posting order, but the Igather result, sent to the root
       only, can overtake the Iallreduce broadcast.) */
    if (parts & 8) {
      long gotWorld = -1, gotPair = -1;
      for (int i = 0; i < paritySize; i++) gathered[i] = -1;
      MPI_Request reqs[3];
      MPI_Iallreduce(&mine, &gotWorld, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &reqs[0]);
      MPI_Iallreduce(&mine, &gotPair, 1, MPI_LONG, MPI_SUM, pair, &reqs[1]);
      MPI_Igather(&mine, 1, MPI_LONG, gathered.data(), 1, MPI_LONG, 0, parity, &reqs[2]);
      MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
      check(rank, iter, "B Iallreduce(world)", gotWorld, wantWorld);
      check(rank, iter, "B Iallreduce(pair)", gotPair, wantPair);
      if (parityRank == 0)
        for (int i = 0; i < paritySize; i++)
          check(rank, iter, "B Igather(parity)[i]", gathered[i], rank % 2 + 2 * i);
    }
  }

  int totalFailures = 0;
  MPI_Allreduce(&failures, &totalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&pair);
  MPI_Comm_free(&parity);
  if (rank == 0) {
    if (totalFailures == 0)
      printf("collwake: all %d iterations passed on %d ranks\n", iters, size);
    else
      printf("collwake: %d failed checks\n", totalFailures);
    fflush(stdout);
  }
  MPI_Finalize();
  return totalFailures == 0 ? 0 : 1;
}
