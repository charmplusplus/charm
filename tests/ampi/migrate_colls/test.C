/**
 * AMPI_Migrate with non-blocking collectives outstanding, in AMPI's default
 * migration mode (no +ampiMigrateWithCollectives).
 *
 * Allowed, and the results must still be right after the ranks have moved:
 * collectives whose results travel as messages to the receiving rank
 * (MPI_Ialltoall, MPI_Iscatter, MPI_Iscan, MPI_Ireduce and MPI_Igather at the
 * root), and every collective on a communicator other than MPI_COMM_WORLD
 * (here MPI_Ibcast and MPI_Iallreduce on a split communicator). Run with
 * +balancer RotateLB so that every rank moves at each AMPI_Migrate.
 *
 * Refused: MPI_Iallreduce, MPI_Iallgather, MPI_Iallgatherv, MPI_Ibarrier or
 * MPI_Ibcast outstanding on MPI_COMM_WORLD, whose results are broadcasts to the
 * array. With the arguments "abort <name>" the program posts that collective
 * and calls AMPI_Migrate; AMPI must abort with a message naming
 * +ampiMigrateWithCollectives, and returning from AMPI_Migrate is the failure.
 *
 * All buffers are arrays on the stack, which migrates with the rank
 * everywhere; heap memory from operator new is not migratable on every
 * platform (on macOS libc++'s operator new bypasses the isomalloc wrapper).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define MAXR 64 // largest MPI_COMM_WORLD this test handles

static int my_pe()
{
  int flag, *wth;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &wth, &flag);
  return flag ? *wth : -1;
}

static void fail(int rank, const char *what, int got, int want)
{
  printf("[%d] ERROR: %s: got %d, expected %d\n", rank, what, got, want);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void expectAbort(const char *name, int rank, int size)
{
  MPI_Request req;
  int in = rank, out = 0;
  int all[MAXR], counts[MAXR], displs[MAXR];
  for (int i = 0; i < size; i++) { counts[i] = 1; displs[i] = i; }

  if (!strcmp(name, "iallreduce"))
    MPI_Iallreduce(&in, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
  else if (!strcmp(name, "iallgather"))
    MPI_Iallgather(&in, 1, MPI_INT, all, 1, MPI_INT, MPI_COMM_WORLD, &req);
  else if (!strcmp(name, "iallgatherv"))
    MPI_Iallgatherv(&in, 1, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD, &req);
  else if (!strcmp(name, "ibarrier"))
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
  else if (!strcmp(name, "ibcast"))
    MPI_Ibcast(&in, 1, MPI_INT, 0, MPI_COMM_WORLD, &req);
  else {
    printf("unknown collective '%s'\n", name);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  AMPI_Migrate(AMPI_INFO_LB_SYNC);

  printf("[%d] ERROR: AMPI_Migrate returned with MPI_%s outstanding on MPI_COMM_WORLD\n",
         rank, name);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 2);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > MAXR) {
    if (rank == 0) printf("migrate_colls handles at most %d ranks\n", MAXR);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Positional arguments; runtime options start with '+' or '-'.
  const char *pos[2] = {NULL, NULL};
  int npos = 0;
  for (int a = 1; a < argc && npos < 2; a++)
    if (argv[a][0] != '+' && argv[a][0] != '-') pos[npos++] = argv[a];
  if (npos == 2 && !strcmp(pos[0], "abort")) expectAbort(pos[1], rank, size);

  const int iters = 3;
  int migrations = 0;

  for (int iter = 0; iter < iters; iter++) {
    const int color = rank % 2;
    MPI_Comm split;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &split);
    int srank, ssize;
    MPI_Comm_rank(split, &srank);
    MPI_Comm_size(split, &ssize);

    MPI_Request reqs[7];
    int nreqs = 0;

    // Ialltoall on the split communicator: results travel as messages.
    int a2aSend[MAXR], a2aRecv[MAXR];
    for (int j = 0; j < ssize; j++) { a2aSend[j] = rank * 1000 + j * 10 + iter; a2aRecv[j] = -1; }
    MPI_Ialltoall(a2aSend, 1, MPI_INT, a2aRecv, 1, MPI_INT, split, &reqs[nreqs++]);

    // Ireduce to rank 0 of MPI_COMM_WORLD: the root's result is a message to
    // its own element.
    int redIn = rank + iter, redOut = -1;
    MPI_Ireduce(&redIn, &redOut, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &reqs[nreqs++]);

    // Igather to the last rank of MPI_COMM_WORLD.
    const int gRoot = size - 1;
    int gRecv[MAXR];
    for (int i = 0; i < size; i++) gRecv[i] = -1;
    MPI_Igather(&rank, 1, MPI_INT, gRecv, 1, MPI_INT, gRoot, MPI_COMM_WORLD, &reqs[nreqs++]);

    // Iscan on MPI_COMM_WORLD: round messages between partner ranks.
    int scanIn = rank + 1, scanOut = -1;
    MPI_Iscan(&scanIn, &scanOut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &reqs[nreqs++]);

    // Iscatter from rank 0 of MPI_COMM_WORLD.
    int scSend[MAXR], scRecv = -1;
    for (int i = 0; i < size; i++) scSend[i] = i * 10 + iter;
    MPI_Iscatter(scSend, 1, MPI_INT, &scRecv, 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[nreqs++]);

    // Ibcast and Iallreduce on the split communicator: their results are
    // broadcasts, but to an array other than MPI_COMM_WORLD's, whose
    // broadcasts the runtime stores for a moving rank.
    int bc = (srank == 0) ? 777 + iter + color : -1;
    MPI_Ibcast(&bc, 1, MPI_INT, 0, split, &reqs[nreqs++]);
    int arIn = rank, arOut = -1;
    MPI_Iallreduce(&arIn, &arOut, 1, MPI_INT, MPI_SUM, split, &reqs[nreqs++]);

    const int peBefore = my_pe();
    AMPI_Migrate(AMPI_INFO_LB_SYNC);
    if (my_pe() != peBefore) migrations++;

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Sub-rank j of color c is world rank 2*j + c.
    for (int j = 0; j < ssize; j++) {
      const int want = (2 * j + color) * 1000 + srank * 10 + iter;
      if (a2aRecv[j] != want) fail(rank, "Ialltoall on a split communicator", a2aRecv[j], want);
    }
    if (rank == 0) {
      const int want = size * (size - 1) / 2 + size * iter;
      if (redOut != want) fail(rank, "Ireduce on MPI_COMM_WORLD", redOut, want);
    }
    if (rank == gRoot)
      for (int i = 0; i < size; i++)
        if (gRecv[i] != i) fail(rank, "Igather on MPI_COMM_WORLD", gRecv[i], i);
    {
      const int want = (rank + 1) * (rank + 2) / 2;
      if (scanOut != want) fail(rank, "Iscan on MPI_COMM_WORLD", scanOut, want);
    }
    {
      const int want = rank * 10 + iter;
      if (scRecv != want) fail(rank, "Iscatter on MPI_COMM_WORLD", scRecv, want);
    }
    {
      const int want = 777 + iter + color;
      if (bc != want) fail(rank, "Ibcast on a split communicator", bc, want);
    }
    {
      int want = 0;
      for (int i = color; i < size; i += 2) want += i;
      if (arOut != want) fail(rank, "Iallreduce on a split communicator", arOut, want);
    }

    MPI_Comm_free(&split);
  }

  int totalMigrations = 0;
  MPI_Reduce(&migrations, &totalMigrations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%d rank migrations over %d iterations\n", totalMigrations, iters);
    printf("All tests passed\n");
  }

  MPI_Finalize();
  return 0;
}
