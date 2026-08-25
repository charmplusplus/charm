/* AMPI under shrink/expand: a 1-D Jacobi iteration that offers every iteration
 * boundary as a point where the job may change width.
 *
 * The contract is two calls. AMPI_Rescale_check() asks, at each boundary,
 * whether a rescale is pending; when it answers yes the rank quiesces and calls
 * AMPI_Migrate(), which is where AMPI ranks were always allowed to move. The
 * runtime agrees on a single iteration across every rank, so they all enter the
 * barrier having completed the same amount of work.
 *
 * Nothing else changes. MPI_COMM_WORLD keeps its size and every rank keeps its
 * number while the number of processes underneath them goes up and down; the
 * checks below are what proves it.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Everything a rank owns lives on the heap or the stack, so that this test
 * needs no globals privatization to be meaningful: both travel with the rank
 * through its Isomalloc context. */
typedef struct {
  int rank, size;
  int n;            /* interior points owned by this rank */
  double *cur;      /* n + 2 halo cells */
  double *next;
  int left, right;
} Domain;

static double boundaryValue(int rank) { return (double)(rank + 1); }

static void domainInit(Domain *d, int rank, int size, int n) {
  d->rank = rank;
  d->size = size;
  d->n = n;
  d->cur = (double *)malloc((n + 2) * sizeof(double));
  d->next = (double *)malloc((n + 2) * sizeof(double));
  if (d->cur == NULL || d->next == NULL) {
    printf("rank %d: out of memory\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < n + 2; i++) d->cur[i] = boundaryValue(rank);
  memcpy(d->next, d->cur, (n + 2) * sizeof(double));
  d->left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  d->right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
}

static void exchangeAndUpdate(Domain *d) {
  MPI_Request req[4];
  int nreq = 0;

  MPI_Irecv(&d->cur[0], 1, MPI_DOUBLE, d->left, 0, MPI_COMM_WORLD, &req[nreq++]);
  MPI_Irecv(&d->cur[d->n + 1], 1, MPI_DOUBLE, d->right, 0, MPI_COMM_WORLD, &req[nreq++]);
  MPI_Isend(&d->cur[1], 1, MPI_DOUBLE, d->left, 0, MPI_COMM_WORLD, &req[nreq++]);
  MPI_Isend(&d->cur[d->n], 1, MPI_DOUBLE, d->right, 0, MPI_COMM_WORLD, &req[nreq++]);
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

  for (int i = 1; i <= d->n; i++)
    d->next[i] = (d->cur[i - 1] + d->cur[i] + d->cur[i + 1]) / 3.0;

  double *tmp = d->cur;
  d->cur = d->next;
  d->next = tmp;
}

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = (argc > 1) ? atoi(argv[1]) : 4096;
  const int maxIter = (argc > 2) ? atoi(argv[2]) : 1000000;
  const int lbPeriod = (argc > 3) ? atoi(argv[3]) : 200;
  const int printEvery = (argc > 4) ? atoi(argv[4]) : 100;

  /* The size and rank this job started with. Neither may ever change, however
   * many processes the job is running on -- that invariant is the point. */
  const int worldSize = size;
  const int myRank = rank;

  Domain d;
  domainInit(&d, rank, size, n);

  MPI_Info hints;
  MPI_Info_create(&hints);
  MPI_Info_set(hints, "ampi_load_balance", "sync");

  double t0 = MPI_Wtime();
  int rescales = 0;

  for (int iter = 1; iter <= maxIter; iter++) {
    exchangeAndUpdate(&d);

    /* A collective every iteration: reductions spanning a rescale are the
     * most delicate part of the runtime, so exercise one continuously. */
    double local = 0.0;
    for (int i = 1; i <= d.n; i++) local += d.cur[i];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Identity checks. A rank that came back with someone else's rank, a
     * different world, or a heap that did not travel would show up here rather
     * than as a wrong answer ten thousand iterations later. */
    int nowRank, nowSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &nowRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nowSize);
    if (nowRank != myRank || nowSize != worldSize) {
      printf("FAIL: rank %d iter %d: world changed under the application "
             "(rank %d, size %d)\n", myRank, iter, nowRank, nowSize);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!(global > 0.0) || global != global /* NaN */) {
      printf("FAIL: rank %d iter %d: allreduce produced %g\n", myRank, iter, global);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* The rescale point. False at every boundary but the agreed one, at the
     * cost of this comparison. */
    int rescaleNow = 0;
    AMPI_Rescale_check(iter, &rescaleNow);

    if (rescaleNow) {
      if (rank == 0)
        printf("rank 0: rescale point at iteration %d\n", iter);
      fflush(stdout);
      AMPI_Migrate(hints);
      rescales++;
    } else if (iter % lbPeriod == 0) {
      AMPI_Migrate(hints);
    }

    if (rank == 0 && iter % printEvery == 0) {
      printf("iteration %d  sum %.6f  elapsed %.3fs  rescale points taken %d\n",
             iter, global, MPI_Wtime() - t0, rescales);
      fflush(stdout);
    }
  }

  MPI_Info_free(&hints);
  free(d.cur);
  free(d.next);
  MPI_Finalize();
  return 0;
}
