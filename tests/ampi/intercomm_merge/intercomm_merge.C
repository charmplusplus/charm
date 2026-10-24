// Test MPI_Intercomm_merge (charm issue #4072, item 1).
// Split COMM_WORLD in two halves, join them with MPI_Intercomm_create, merge
// the intercommunicator with every combination of the `high` argument, and
// check the size, the rank order and an Allreduce on the merged communicator.
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <atomic>
#include "mpi.h"

static int errors = 0;

// Shared by all ranks of one process on purpose (not privatized): counts the
// members of a new communicator that have entered the barrier after its creation.
static std::atomic<int> membersInBarrier{0};

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
    if (rank == 0) printf("intercomm_merge needs at least 2 ranks, skipping\n");
    MPI_Finalize();
    return 0;
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

  // Ranks that are not members of a communicator created from a subset must
  // not be resumed by that creation (they return MPI_COMM_NULL at once and
  // may block in something else, here a barrier that the members can only
  // enter once the creation is complete).
  {
    int *numProcs = nullptr, flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_NUM_PROCESSES, &numProcs, &flag);
    const bool oneProcess = flag && *numProcs == 1;
    for (int round = 0; round < 3; round++) {
      MPI_Comm cart;
      int dims[1] = { size - 1 }, periods[1] = { 0 };
      MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart);
      const bool member = rank < size - 1;
      check(member == (cart != MPI_COMM_NULL), "cart membership", rank, 4 + round);
      if (member) membersInBarrier++;
      MPI_Barrier(MPI_COMM_WORLD);
      if (oneProcess)
        check(membersInBarrier.load() == (size - 1) * (round + 1), "barrier waited for all members", rank, 4 + round);
      MPI_Barrier(MPI_COMM_WORLD);
      if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    }
    // A 0-dimensional Cartesian communicator has rank 0 as its only member
    membersInBarrier = 0;
    MPI_Comm cart0;
    MPI_Cart_create(MPI_COMM_WORLD, 0, nullptr, nullptr, 0, &cart0);
    check((rank == 0) == (cart0 != MPI_COMM_NULL), "0-d cart membership", rank, 7);
    if (cart0 != MPI_COMM_NULL) {
      int csize = -1;
      MPI_Comm_size(cart0, &csize);
      check(csize == 1, "0-d cart size", rank, 7);
      membersInBarrier++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (oneProcess)
      check(membersInBarrier.load() == 1, "barrier waited for the 0-d cart member", rank, 7);
    MPI_Barrier(MPI_COMM_WORLD);
    if (cart0 != MPI_COMM_NULL) MPI_Comm_free(&cart0);
  }

  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0) printf("intercomm_merge: All tests passed\n");
    else printf("intercomm_merge: %d errors\n", total);
  }
  // AMPI's MPI_Finalize never returns (it ends in AMPI_Exit(0)), so a
  // failure must be reported through the exit status of MPI_Abort.
  if (total != 0) MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
  return 0;
}
