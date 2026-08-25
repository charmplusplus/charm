/* AMPI under shrink/expand: several reduction trees in flight at once.
 *
 * The companion test, jacobi1d.c, reduces over MPI_COMM_WORLD and nothing
 * else, so exactly one reduction tree crosses each cut. That is the easy case
 * and it is the only case the campaigns had covered.
 *
 * Every AMPI communicator is its own CkArray with its own CkReductionMgr, and
 * therefore its own spanning tree, its own round counter, and its own opinion
 * about which PEs are contributing. A rescale rebuilds all of them at once.
 * RESCALE_KNOWN_ISSUES.md R3 says only the first post-rescale round is
 * tolerated and that several rounds in flight have no tolerance at all, so
 * this is where a post-rescale hang would be expected to come from.
 *
 * So: four trees per iteration, deliberately different shapes.
 *
 *   world   every rank; the shape the existing tests already cover
 *   row     ranks split into ROWS groups by rank / rowSize
 *   col     ranks split into rowSize groups by rank % rowSize -- interleaved,
 *           so a row and a column group share no PE layout in common
 *   half    only even ranks take part; odd ranks are outside it entirely,
 *           which leaves PEs holding no contributor for this tree and
 *           exercises the inactive-list path rather than the counting one
 *
 * The row and column reductions are issued non-blocking and waited on
 * together, so all four rounds genuinely overlap rather than running one after
 * another. Everything is quiesced before the rescale point: a rank parked in
 * AMPI_Migrate with a collective outstanding is a different question, and one
 * hold-boundary mode exists to avoid.
 *
 * Each reduction has an exact answer that depends only on rank numbers, so a
 * tree that loses or double-counts a contribution is caught on the iteration
 * it happens rather than showing up as a slow drift.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* On the heap and the stack only, so no globals privatization is needed for
 * the test to mean anything -- both travel with the rank. */
typedef struct {
  int rank, size;
  int rowSize;              /* ranks per row */
  MPI_Comm row, col, half;
  int rowRank, rowMembers;
  int colRank, colMembers;
  int inHalf, halfMembers;
  int n;
  double *work;
} World;

/* Sum of rank numbers over a contiguous or strided set, computed directly so
 * the check does not depend on the reduction it is checking. */
static double sumOver(int first, int stride, int count) {
  double s = 0.0;
  for (int i = 0; i < count; i++) s += (double)(first + i * stride);
  return s;
}

static void worldInit(World *w, int rank, int size, int rowSize, int n) {
  w->rank = rank;
  w->size = size;
  w->rowSize = rowSize;
  w->n = n;

  w->work = (double *)malloc((size_t)n * sizeof(double));
  if (w->work == NULL) {
    printf("rank %d: out of memory\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < n; i++) w->work[i] = (double)(rank + 1);

  /* Rows are contiguous blocks of rowSize ranks; columns take every rowSize-th
     rank. Two ranks that share a row share no column, which is the point --
     the two trees have genuinely different membership. */
  MPI_Comm_split(MPI_COMM_WORLD, rank / rowSize, rank, &w->row);
  MPI_Comm_split(MPI_COMM_WORLD, rank % rowSize, rank, &w->col);
  /* MPI_UNDEFINED leaves the odd ranks out of the communicator altogether,
     rather than giving them one of their own. */
  MPI_Comm_split(MPI_COMM_WORLD, (rank % 2 == 0) ? 0 : MPI_UNDEFINED, rank,
                 &w->half);

  MPI_Comm_rank(w->row, &w->rowRank);
  MPI_Comm_size(w->row, &w->rowMembers);
  MPI_Comm_rank(w->col, &w->colRank);
  MPI_Comm_size(w->col, &w->colMembers);

  w->inHalf = (w->half != MPI_COMM_NULL);
  w->halfMembers = 0;
  if (w->inHalf) MPI_Comm_size(w->half, &w->halfMembers);
}

/* What each tree must produce, from rank numbers alone. */
static double expectWorld(const World *w) { return sumOver(0, 1, w->size); }

static double expectRow(const World *w) {
  const int first = (w->rank / w->rowSize) * w->rowSize;
  return sumOver(first, 1, w->rowMembers);
}

static double expectCol(const World *w) {
  const int first = w->rank % w->rowSize;
  return sumOver(first, w->rowSize, w->colMembers);
}

static double expectHalf(const World *w) { return sumOver(0, 2, w->halfMembers); }

static int check(const World *w, int iter, const char *what, double got,
                 double want) {
  /* Exact: these are sums of small integers in doubles. A tree that dropped a
     contribution or counted one twice is off by at least one whole rank. */
  if (got != want) {
    printf("FAIL: rank %d iter %d: %s reduced to %.1f, expected %.1f\n",
           w->rank, iter, what, got, want);
    fflush(stdout);
    return 0;
  }
  return 1;
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
  int rowSize = (argc > 5) ? atoi(argv[5]) : 4;
  /* 1: row/col reductions overlap the world one. 0: each finishes before the
     next starts, which is the same four trees without the concurrency. */
  const int overlap = (argc > 6) ? atoi(argv[6]) : 1;

  if (rowSize < 1) rowSize = 1;
  if (rowSize > size) rowSize = size;

  /* The size and rank this job started with. Neither may ever change, however
   * many processes the job is running on. */
  const int worldSize = size;
  const int myRank = rank;

  World w;
  worldInit(&w, rank, size, rowSize, n);

  if (rank == 0) {
    printf("subcomm: %d ranks, rowSize %d -> %d rows of %d, columns of %d, "
           "half-comm of %d\n",
           size, rowSize, (size + rowSize - 1) / rowSize, w.rowMembers,
           w.colMembers, w.halfMembers);
    fflush(stdout);
  }

  MPI_Info hints;
  MPI_Info_create(&hints);
  MPI_Info_set(hints, "ampi_load_balance", "sync");

  double t0 = MPI_Wtime();
  int rescales = 0;

  for (int iter = 1; iter <= maxIter; iter++) {
    /* Some work, so the load balancer has something to weigh. */
    double local = 0.0;
    for (int i = 0; i < w.n; i++) {
      w.work[i] = (w.work[i] + (double)(w.rank + 1)) * 0.5;
      local += w.work[i];
    }
    (void)local;

    const double mine = (double)w.rank;
    double gotWorld = 0.0, gotRow = 0.0, gotCol = 0.0, gotHalf = 0.0;

    /* Row and column go out first and are not waited on yet, so their rounds
       overlap each other and the world round below. */
    MPI_Request req[2];
    MPI_Iallreduce(&mine, &gotRow, 1, MPI_DOUBLE, MPI_SUM, w.row, &req[0]);
    MPI_Iallreduce(&mine, &gotCol, 1, MPI_DOUBLE, MPI_SUM, w.col, &req[1]);
    if (!overlap) MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    MPI_Allreduce(&mine, &gotWorld, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (w.inHalf)
      MPI_Allreduce(&mine, &gotHalf, 1, MPI_DOUBLE, MPI_SUM, w.half);

    if (overlap) MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    /* Identity first: a rank that came back as someone else, or into a
       different world, would make every reduction check meaningless. */
    int nowRank, nowSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &nowRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nowSize);
    if (nowRank != myRank || nowSize != worldSize) {
      printf("FAIL: rank %d iter %d: world changed under the application "
             "(rank %d, size %d)\n", myRank, iter, nowRank, nowSize);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ok = 1;
    ok &= check(&w, iter, "MPI_COMM_WORLD", gotWorld, expectWorld(&w));
    ok &= check(&w, iter, "row", gotRow, expectRow(&w));
    ok &= check(&w, iter, "col", gotCol, expectCol(&w));
    if (w.inHalf) ok &= check(&w, iter, "half", gotHalf, expectHalf(&w));
    if (!ok) MPI_Abort(MPI_COMM_WORLD, 1);

    /* Every tree is quiesced here: the rescale lands between rounds, not
       inside one. */
    int rescaleNow = 0;
    AMPI_Rescale_check(iter, &rescaleNow);

    if (rescaleNow) {
      if (rank == 0) printf("rank 0: rescale point at iteration %d\n", iter);
      fflush(stdout);
      AMPI_Migrate(hints);
      rescales++;
    } else if (iter % lbPeriod == 0) {
      AMPI_Migrate(hints);
    }

    if (rank == 0 && iter % printEvery == 0) {
      printf("iteration %d  world %.1f  row %.1f  col %.1f  elapsed %.3fs  "
             "rescale points taken %d\n",
             iter, gotWorld, gotRow, gotCol, MPI_Wtime() - t0, rescales);
      fflush(stdout);
    }
  }

  MPI_Info_free(&hints);
  if (w.row != MPI_COMM_NULL) MPI_Comm_free(&w.row);
  if (w.col != MPI_COMM_NULL) MPI_Comm_free(&w.col);
  if (w.half != MPI_COMM_NULL) MPI_Comm_free(&w.half);
  free(w.work);
  MPI_Finalize();
  return 0;
}
