/* In-memory checkpoint and rollback of AMPI ranks (ampi_checkpoint=in_memory
   and ampi_checkpoint=rollback; needs a build with the syncft option).

   Every rank keeps a stack array and a heap buffer whose contents depend on
   the iteration, and a heap structure (struct rank_state) with the running
   sum of the global reductions. Each of iterations 1..10 checks that both
   arrays still hold the previous iteration's values, rewrites them, does an
   MPI_Allreduce and a point-to-point exchange with the neighbor ranks, and
   adds the reduction to the sum.

   At iteration 5 every rank takes an in-memory checkpoint. At iteration 8,
   if AMPI_Rollback_count reports no rollback yet, every rank asks for a
   rollback. The rollback restores every rank: the thread returns from the
   iteration-5 AMPI_Migrate a second time, with the stack and the heap as
   they were when the checkpoint was taken, and iterations 6-10 run again.

   Telling the two returns apart: a counter in the rank's state cannot do
   it. The checkpoint is taken while the thread is suspended inside the
   iteration-5 AMPI_Migrate, so every rollback restores the counter to its
   value before the first return. The test shows this: `returns` in the
   heap is incremented after each return and is 1 both times.
   AMPI_Rollback_count gives the count kept by the runtime: 0 after the
   first return, 1 after the return caused by the rollback.

   Witnesses that iterations 6-8 ran twice and from the checkpointed state:
   iteration 8 is reached with count 0 (the rollback is requested there, and
   the line after the rollback call, which must never run, fails the test);
   after the second return the count is 1, and the check of the arrays
   right after that return finds the iteration-5 values although the
   rollback was requested when they held the iteration-8 values; rank 0
   prints "iteration I pass P" for each iteration, so the output shows 6, 7
   and 8 under pass 1 and again under pass 2; the final sum must equal the
   closed form of a run with no rollback, and the final count must be 1.

   Options:
     -info        pass an MPI_Info built with MPI_Info_set instead of the
                  built-in objects AMPI_INFO_CHKPT_IN_MEMORY and
                  AMPI_INFO_CHKPT_ROLLBACK
     -second      after the rollback, take a second in-memory checkpoint at
                  iteration 9 and roll back to it at iteration 10, once:
                  the checkpoint buffers are reused, and the count starts
                  again from 0 at the new checkpoint
     -norollback  checkpoint only (the plain in_memory path) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define NITER 10
#define CKPT_ITER 5
#define ROLLBACK_ITER 8
#define CKPT2_ITER 9
#define ROLLBACK2_ITER 10
#define NSTACK 16
#define NHEAP 1000

struct rank_state {
  int returns;  /* returns from a checkpoint call: restored by a rollback */
  long sum;     /* running sum of the global reductions */
  int *buf;     /* NHEAP values, rewritten every iteration */
};

static int value(int rank, int iter, int i) { return rank * 1000 + iter * 37 + i; }

static void fail(int rank, const char *what, int iter)
{
  printf("[%d] FAIL at iteration %d: %s\n", rank, iter, what);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void check_arrays(int rank, int iter, const int *stk, const int *heap)
{
  for (int i = 0; i < NSTACK; i++)
    if (stk[i] != value(rank, iter, i)) fail(rank, "stack array has a wrong value", iter);
  for (int i = 0; i < NHEAP; i++)
    if (heap[i] != value(rank, iter, i)) fail(rank, "heap buffer has a wrong value", iter);
}

int main(int argc, char **argv)
{
  int rank, size, useInfo = 0, second = 0, norollback = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int a = 1; a < argc; a++) {
    if (strcmp(argv[a], "-info") == 0) useInfo = 1;
    else if (strcmp(argv[a], "-second") == 0) second = 1;
    else if (strcmp(argv[a], "-norollback") == 0) norollback = 1;
  }

  MPI_Info ckptInfo = AMPI_INFO_CHKPT_IN_MEMORY, rollbackInfo = AMPI_INFO_CHKPT_ROLLBACK;
  if (useInfo) {
    MPI_Info_create(&ckptInfo);
    MPI_Info_set(ckptInfo, "ampi_checkpoint", "in_memory");
    MPI_Info_create(&rollbackInfo);
    MPI_Info_set(rollbackInfo, "ampi_checkpoint", "rollback");
  }

  struct rank_state *st = (struct rank_state *)malloc(sizeof(struct rank_state));
  st->returns = 0;
  st->sum = 0;
  st->buf = (int *)malloc(NHEAP * sizeof(int));
  int stk[NSTACK];
  for (int i = 0; i < NSTACK; i++) stk[i] = value(rank, 0, i);
  for (int i = 0; i < NHEAP; i++) st->buf[i] = value(rank, 0, i);

  int count = 0; /* AMPI_Rollback_count after the last checkpoint call */
  const int right = (rank + 1) % size, left = (rank + size - 1) % size;
  for (int iter = 1; iter <= NITER; iter++) {
    check_arrays(rank, iter - 1, stk, st->buf);
    for (int i = 0; i < NSTACK; i++) stk[i] = value(rank, iter, i);
    for (int i = 0; i < NHEAP; i++) st->buf[i] = value(rank, iter, i);
    if (rank == 0) {
      printf("iteration %d pass %d\n", iter, count + 1);
      fflush(stdout);
    }

    long local = st->buf[iter % NHEAP] + stk[iter % NSTACK], global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    st->sum += global;

    int out = value(rank, iter, -1), in = -1;
    MPI_Sendrecv(&out, 1, MPI_INT, right, iter, &in, 1, MPI_INT, left, iter,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (in != value(left, iter, -1)) fail(rank, "wrong value from the left neighbor", iter);

    if (iter == CKPT_ITER || (second && iter == CKPT2_ITER)) {
      st->returns = 0;
      AMPI_Migrate(ckptInfo);
      st->returns++;
      AMPI_Rollback_count(&count);
      check_arrays(rank, iter, stk, st->buf);
      if (st->returns != 1) fail(rank, "the rollback did not restore the heap counter", iter);
      if (rank == 0) {
        printf("returned from the iteration-%d checkpoint, rollbacks %d\n", iter, count);
        fflush(stdout);
      }
    }
    if (((!norollback && iter == ROLLBACK_ITER) || (second && iter == ROLLBACK2_ITER))
        && count == 0) {
      AMPI_Migrate(rollbackInfo);
      fail(rank, "AMPI_Migrate(rollback) returned", iter);
    }
  }

  /* The sum a run without rollback computes. */
  long expected = 0;
  for (int iter = 1; iter <= NITER; iter++)
    for (int r = 0; r < size; r++)
      expected += value(r, iter, iter % NHEAP) + value(r, iter, iter % NSTACK);
  if (st->sum != expected) fail(rank, "final sum differs from a run without rollback", NITER);
  if (count != (norollback ? 0 : 1)) fail(rank, "wrong rollback count at the end", NITER);

  if (useInfo) {
    MPI_Info_free(&ckptInfo);
    MPI_Info_free(&rollbackInfo);
  }
  free(st->buf);
  free(st);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("All tests passed\n");
  MPI_Finalize();
  return 0;
}
