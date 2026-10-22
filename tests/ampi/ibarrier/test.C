// MPI_Ibarrier completion and one-sided replies built from AmpiMsg.
//
// ampi::sendraw (used by MPI_Ibarrier) and the one-sided reply paths in
// ampiOneSided.C built their AmpiMsg with a plain int 0 as the first
// argument, which selected the wrong constructor and shifted every field.
// MPI_Ibarrier + MPI_Wait then hung with more than one rank.
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

static int errors = 0;

static void check(bool ok, int rank, const char* what, int got, int want)
{
  if (!ok) {
    printf("[%d] FAIL %s: got %d, expected %d\n", rank, what, got, want);
    errors++;
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 1. MPI_Ibarrier + MPI_Wait, several times.
  for (int i = 0; i < 5; i++) {
    MPI_Request req;
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }
  if (rank == 0) printf("ibarrier: MPI_Ibarrier + MPI_Wait x5 done on %d ranks\n", size);

  // 2. One-sided operations. When the target rank lives on another PE these
  // take the reply-message paths (ampi::winRemoteGet,
  // ampi::winRemoteGetAccumulate), which build their reply with
  // AmpiMsg(0, 0, MPI_RMA_TAG, ...). AMPI RMA supports only basic datatypes.
  int wbuf[4];
  for (int i = 0; i < 4; i++) wbuf[i] = rank * 10 + i;
  MPI_Win win;
  MPI_Win_create(wbuf, sizeof(wbuf), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  int target = (rank + 1) % size;

  int gbuf[2] = {-1, -1};
  MPI_Win_fence(0, win);
  MPI_Get(gbuf, 2, MPI_INT, target, 2, 2, MPI_INT, win);
  MPI_Win_fence(0, win);
  check(gbuf[0] == target * 10 + 2, rank, "get[0]", gbuf[0], target * 10 + 2);
  check(gbuf[1] == target * 10 + 3, rank, "get[1]", gbuf[1], target * 10 + 3);

  int obuf[2] = {1, 1};
  int rbuf[2] = {-1, -1};
  MPI_Win_fence(0, win);
  MPI_Get_accumulate(obuf, 2, MPI_INT, rbuf, 2, MPI_INT, target, 1, 2, MPI_INT, MPI_SUM, win);
  MPI_Win_fence(0, win);
  check(rbuf[0] == target * 10 + 1, rank, "get_accumulate result[0]", rbuf[0], target * 10 + 1);
  check(rbuf[1] == target * 10 + 2, rank, "get_accumulate result[1]", rbuf[1], target * 10 + 2);
  check(wbuf[1] == rank * 10 + 2, rank, "accumulated wbuf[1]", wbuf[1], rank * 10 + 2);
  check(wbuf[2] == rank * 10 + 3, rank, "accumulated wbuf[2]", wbuf[2], rank * 10 + 3);

  MPI_Win_free(&win);

  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0) printf("ibarrier test passed on %d ranks\n", size);
    else printf("ibarrier test FAILED: %d errors\n", total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
