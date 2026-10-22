// Test MPI_Intercomm_merge (charm issue #4072, item 1).
// Split COMM_WORLD in two halves, join them with MPI_Intercomm_create, merge
// the intercommunicator with every combination of the `high` argument, and
// check the size, the rank order and an Allreduce on the merged communicator.
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mpi.h"

static int errors = 0;

static void check(bool cond, const char *what, int rank, int round) {
  if (!cond) {
    printf("[%d] round %d: FAILED %s\n", rank, round, what);
    errors++;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    if (rank == 0) printf("intercomm_merge needs at least 2 ranks\n");
    MPI_Finalize();
    return 1;
  }

  const int half = size / 2;
  const int color = (rank < half) ? 0 : 1;
  const int lsize = color ? size - half : half;

  // high arguments for (group 0, group 1); the group passing high=0 comes
  // first; on a tie the group whose root has the lower index comes first.
  const int highs[4][2] = { {0, 1}, {1, 0}, {0, 0}, {1, 1} };

  for (int round = 0; round < 4; round++) {
    MPI_Comm local, inter, merged;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &local);
    MPI_Intercomm_create(local, 0, MPI_COMM_WORLD, color ? 0 : half, 99, &inter);

    int rsize = -1;
    MPI_Comm_remote_size(inter, &rsize);
    check(rsize == size - lsize, "remote size", rank, round);

    const int high = highs[round][color];
    MPI_Intercomm_merge(inter, high, &merged);

    int msize = -1, mrank = -1;
    MPI_Comm_size(merged, &msize);
    MPI_Comm_rank(merged, &mrank);
    check(msize == size, "merged size", rank, round);

    const bool group0First = (highs[round][0] == highs[round][1]) || highs[round][0] == 0;
    const int lrank = color ? rank - half : rank;
    int expect;
    if (group0First) expect = rank;
    else expect = color ? lrank : (size - half) + lrank;
    check(mrank == expect, "merged rank", rank, round);

    // Rank order as seen by everyone
    std::vector<int> order(size, -1);
    MPI_Allgather(&rank, 1, MPI_INT, order.data(), 1, MPI_INT, merged);
    for (int i = 0; i < size; i++) {
      int w = group0First ? i : (i < size - half ? half + i : i - (size - half));
      check(order[i] == w, "allgather order", rank, round);
    }

    int sum = -1;
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, merged);
    check(sum == size * (size - 1) / 2, "allreduce sum", rank, round);

    MPI_Comm_free(&merged);
    MPI_Comm_free(&inter);
    MPI_Comm_free(&local);
  }

  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0) printf("intercomm_merge: All tests passed\n");
    else printf("intercomm_merge: %d errors\n", total);
  }
  MPI_Finalize();
  return total != 0;
}
