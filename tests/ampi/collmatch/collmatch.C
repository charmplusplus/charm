/* Matching of non-blocking collective results to their requests
   (ampi::irednResult; issue #4072, item 7).

   Several non-blocking reductions/gathers are outstanding on the same
   communicator. Their results can reach a rank in any order: an MPI_Igather
   result goes only to the root and can arrive before the broadcast of an
   earlier MPI_Iallreduce. Each result must land in its own request.

   Part 1: Iallreduce then Igather on MPI_COMM_WORLD, waited in both orders.
   Part 2: Igather then Iallreduce, waited in both orders.
   Part 3: Iallreduce, Igather, Ireduce, Iallreduce of a different length,
           and Iallgather, all outstanding, completed with MPI_Waitall.
   Part 4: the same set on a sub-communicator while MPI_COMM_WORLD has one
           Iallreduce outstanding.
   Part 5: an MPI_Irecv(MPI_ANY_SOURCE, MPI_ANY_TAG) is posted before an
           Iallreduce on the same communicator; the reduction result must not
           be handed to the receive (it used to be: the result was matched by
           (tag, source) in the same queue, and the wildcard receive matched).
   Part 6: MPI_Ibarrier interleaved with MPI_Ialltoall on MPI_COMM_WORLD
           (barrier first and alltoall first), and an Ibarrier behind a
           wildcard Irecv. The barrier used to complete through a message
           keyed like rank 0's alltoall block.
   Part 7: MPI_Igatherv (rank r sends r+1 values) with NULL recvcounts and
           displs on the non-roots, which MPI allows, outstanding together
           with an Iallreduce.
   Part 8 (every 10th iteration): one of Iallreduce, Iallgatherv (rank r
           sends r+1 values), Ibarrier, Igather on MPI_COMM_WORLD, rotating
           per stage, posted by every rank before AMPI_Migrate and waited on
           after it, so the request is outstanding (posted under its key)
           while load balancing runs. With +balancer RandCentLB on more than
           one PE ranks migrate, which exercises the pup path. Every rank
           posts before AMPI_Migrate because AMPI_Migrate is itself
           collective: MPI requires all ranks to call collectives on a
           communicator in the same order, so an earlier version that had
           the odd ranks post after AMPI_Migrate was not a valid program
           (it aborted intermittently with "recv'ed a non-blocking reduction
           unexpectedly", as classic AMPI does for an MPI_Barrier placed in
           that order). Exactly one collective is kept in flight across
           AMPI_Migrate: with several in flight across the AtSync/CentralLB
           step the job hangs intermittently, a core-side problem tracked
           on charm #4072.

   Roots rotate with the iteration so that every rank is sometimes the root.
   Every result is checked against a value computed locally. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static int failures = 0;

static void check(int rank, int iter, const char *what, long got, long want)
{
  if (got != want) {
    if (failures < 20) {
      printf("FAIL: rank %d iter %d: %s = %ld, expected %ld\n", rank, iter, what, got, want);
      fflush(stdout);
    }
    failures++;
  }
}

/* Values contributed by a rank (rank within the communicator used). */
static long redVal(int r, int iter, int k) { return (long)(r + 1) * (k + 1) + iter; }
static long gatVal(int r, int iter) { return 1000L * r + iter; }

struct Expected {
  int size, iter;
  long sum(int k) const {
    long s = 0;
    for (int r = 0; r < size; r++) s += redVal(r, iter, k);
    return s;
  }
};

/* One set of outstanding collectives on comm (part 3 and 4). */
struct CollSet {
  enum { NLONG = 3 };
  long red1, redLong[NLONG], redRoot;
  std::vector<long> gat, allgat;
  MPI_Request reqs[5];

  void post(MPI_Comm comm, int rank, int size, int iter) {
    const int root = iter % size;
    const int rroot = (iter + size / 2) % size;
    long in1 = redVal(rank, iter, 0), inLong[NLONG], inG = gatVal(rank, iter);
    for (int k = 0; k < NLONG; k++) inLong[k] = redVal(rank, iter, k);
    red1 = -1; redRoot = -1;
    for (int k = 0; k < NLONG; k++) redLong[k] = -1;
    gat.assign(size, -1);
    allgat.assign(size, -1);
    MPI_Iallreduce(&in1, &red1, 1, MPI_LONG, MPI_SUM, comm, &reqs[0]);
    MPI_Igather(&inG, 1, MPI_LONG, gat.data(), 1, MPI_LONG, root, comm, &reqs[1]);
    MPI_Ireduce(&in1, &redRoot, 1, MPI_LONG, MPI_MAX, rroot, comm, &reqs[2]);
    MPI_Iallreduce(inLong, redLong, NLONG, MPI_LONG, MPI_SUM, comm, &reqs[3]);
    MPI_Iallgather(&inG, 1, MPI_LONG, allgat.data(), 1, MPI_LONG, comm, &reqs[4]);
  }

  void verify(const char *tag, int wrank, int rank, int size, int iter) {
    const int root = iter % size;
    const int rroot = (iter + size / 2) % size;
    Expected e{size, iter};
    char what[128];
    snprintf(what, sizeof what, "%s Iallreduce", tag);
    check(wrank, iter, what, red1, e.sum(0));
    for (int k = 0; k < NLONG; k++) {
      snprintf(what, sizeof what, "%s Iallreduce[%d] (length %d)", tag, k, NLONG);
      check(wrank, iter, what, redLong[k], e.sum(k));
    }
    if (rank == root)
      for (int r = 0; r < size; r++) {
        snprintf(what, sizeof what, "%s Igather[%d]", tag, r);
        check(wrank, iter, what, gat[r], gatVal(r, iter));
      }
    if (rank == rroot) {
      snprintf(what, sizeof what, "%s Ireduce(MAX)", tag);
      check(wrank, iter, what, redRoot, redVal(size - 1, iter, 0));
    }
    for (int r = 0; r < size; r++) {
      snprintf(what, sizeof what, "%s Iallgather[%d]", tag, r);
      check(wrank, iter, what, allgat[r], gatVal(r, iter));
    }
  }
};

/* Parts 1 and 2: an Iallreduce and an Igather on comm, posted in the order
   given by gatherFirst, waited in the order given by waitGatherFirst. */
static void pair(MPI_Comm comm, int rank, int size, int iter, bool gatherFirst,
                 bool waitGatherFirst)
{
  const int root = (iter + (gatherFirst ? 1 : 0)) % size;
  long in = redVal(rank, iter, 0), inG = gatVal(rank, iter), red = -1;
  std::vector<long> gat(size, -1);
  MPI_Request rr, rg;
  if (gatherFirst) {
    MPI_Igather(&inG, 1, MPI_LONG, gat.data(), 1, MPI_LONG, root, comm, &rg);
    MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, comm, &rr);
  } else {
    MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, comm, &rr);
    MPI_Igather(&inG, 1, MPI_LONG, gat.data(), 1, MPI_LONG, root, comm, &rg);
  }
  if (waitGatherFirst) {
    MPI_Wait(&rg, MPI_STATUS_IGNORE);
    MPI_Wait(&rr, MPI_STATUS_IGNORE);
  } else {
    MPI_Wait(&rr, MPI_STATUS_IGNORE);
    MPI_Wait(&rg, MPI_STATUS_IGNORE);
  }
  const char *what = gatherFirst ? (waitGatherFirst ? "P2 gg" : "P2 gr")
                                 : (waitGatherFirst ? "P1 rg" : "P1 rr");
  Expected e{size, iter};
  char buf[96];
  snprintf(buf, sizeof buf, "%s Iallreduce", what);
  check(rank, iter, buf, red, e.sum(0));
  if (rank == root)
    for (int r = 0; r < size; r++) {
      snprintf(buf, sizeof buf, "%s Igather[%d]", what, r);
      check(rank, iter, buf, gat[r], gatVal(r, iter));
    }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int iters = (argc > 1) ? atoi(argv[1]) : 50;
  /* Bit mask of the parts to run: 1 = parts 1-4, 2 = part 5, 4 = part 6,
     8 = part 7, 16 = part 8. */
  const int parts = (argc > 2) ? atoi(argv[2]) : 31;
  int moves = 0; /* times this rank changed PE across AMPI_Migrate */

  MPI_Comm sub; /* even ranks / odd ranks */
  MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &sub);
  int subRank, subSize;
  MPI_Comm_rank(sub, &subRank);
  MPI_Comm_size(sub, &subSize);

  for (int iter = 0; iter < iters; iter++) {
    const long in = redVal(rank, iter, 0);
    long red = -1;
    if (parts & 1) {
      /* Parts 1 and 2 */
      pair(MPI_COMM_WORLD, rank, size, iter, false, false);
      pair(MPI_COMM_WORLD, rank, size, iter, false, true);
      pair(MPI_COMM_WORLD, rank, size, iter, true, false);
      pair(MPI_COMM_WORLD, rank, size, iter, true, true);

      /* Part 3 */
      CollSet world;
      world.post(MPI_COMM_WORLD, rank, size, iter);
      MPI_Waitall(5, world.reqs, MPI_STATUSES_IGNORE);
      world.verify("P3 world", rank, rank, size, iter);

      /* Part 4 */
      MPI_Request wreq;
      MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &wreq);
      CollSet s;
      s.post(sub, subRank, subSize, iter);
      MPI_Waitall(5, s.reqs, MPI_STATUSES_IGNORE);
      MPI_Wait(&wreq, MPI_STATUS_IGNORE);
      s.verify("P4 sub", rank, subRank, subSize, iter);
      check(rank, iter, "P4 world Iallreduce", red, Expected{size, iter}.sum(0));
    }

    /* Part 5 */
    if (parts & 2) {
      long msg = -1, out = 1000L * rank + iter;
      MPI_Request rreq, areq;
      MPI_Irecv(&msg, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreq);
      red = -1;
      MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &areq);
      MPI_Wait(&areq, MPI_STATUS_IGNORE);
      check(rank, iter, "P5 Iallreduce behind wildcard Irecv", red, Expected{size, iter}.sum(0));
      MPI_Send(&out, 1, MPI_LONG, (rank + 1) % size, 7, MPI_COMM_WORLD);
      MPI_Wait(&rreq, MPI_STATUS_IGNORE);
      check(rank, iter, "P5 wildcard Irecv", msg, 1000L * ((rank + size - 1) % size) + iter);
      /* A wildcard receive also matches AMPI's internal point-to-point
         traffic (for example Ialltoall blocks; a separate issue), so no rank
         may start the next part while another's wildcard receive is pending. */
      MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Part 6 */
    if (parts & 4) {
      std::vector<long> sendb(size), recvb(size);
      for (int barrierFirst = 0; barrierFirst < 2; barrierFirst++) {
        for (int r = 0; r < size; r++) { sendb[r] = 100000L * rank + 100L * r + iter; recvb[r] = -1; }
        MPI_Request q[2];
        if (barrierFirst) {
          MPI_Ibarrier(MPI_COMM_WORLD, &q[0]);
          MPI_Ialltoall(sendb.data(), 1, MPI_LONG, recvb.data(), 1, MPI_LONG, MPI_COMM_WORLD, &q[1]);
        } else {
          MPI_Ialltoall(sendb.data(), 1, MPI_LONG, recvb.data(), 1, MPI_LONG, MPI_COMM_WORLD, &q[1]);
          MPI_Ibarrier(MPI_COMM_WORLD, &q[0]);
        }
        /* Separate waits: MPI_Waitall on an Ialltoall request hangs, a
           separate pre-existing defect. */
        MPI_Wait(&q[barrierFirst ? 0 : 1], MPI_STATUS_IGNORE);
        MPI_Wait(&q[barrierFirst ? 1 : 0], MPI_STATUS_IGNORE);
        for (int r = 0; r < size; r++) {
          char what[96];
          snprintf(what, sizeof what, "P6 Ialltoall[%d] (%s first)", r, barrierFirst ? "Ibarrier" : "Ialltoall");
          check(rank, iter, what, recvb[r], 100000L * r + 100L * rank + iter);
        }
      }

      long msg = -1, out = 2000L * rank + iter;
      MPI_Request rreq, breq;
      MPI_Irecv(&msg, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreq);
      MPI_Ibarrier(MPI_COMM_WORLD, &breq);
      MPI_Wait(&breq, MPI_STATUS_IGNORE);
      MPI_Send(&out, 1, MPI_LONG, (rank + 1) % size, 8, MPI_COMM_WORLD);
      MPI_Wait(&rreq, MPI_STATUS_IGNORE);
      check(rank, iter, "P6 wildcard Irecv behind Ibarrier", msg, 2000L * ((rank + size - 1) % size) + iter);
      MPI_Barrier(MPI_COMM_WORLD); /* as in part 5 */
    }

    /* Part 7 */
    if (parts & 8) {
      const int root = (iter + 2) % size;
      std::vector<long> sendv(rank + 1);
      for (int k = 0; k <= rank; k++) sendv[k] = gatVal(rank, iter) + 10L * k;
      std::vector<int> counts, displs;
      std::vector<long> recvv;
      if (rank == root) {
        counts.resize(size); displs.resize(size);
        int off = 0;
        for (int r = 0; r < size; r++) { counts[r] = r + 1; displs[r] = off; off += r + 1; }
        recvv.assign(off, -1);
      }
      red = -1;
      MPI_Request q[2];
      MPI_Igatherv(sendv.data(), rank + 1, MPI_LONG, rank == root ? recvv.data() : NULL,
                   rank == root ? counts.data() : NULL, rank == root ? displs.data() : NULL,
                   MPI_LONG, root, MPI_COMM_WORLD, &q[0]);
      MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &q[1]);
      MPI_Waitall(2, q, MPI_STATUSES_IGNORE);
      check(rank, iter, "P7 Iallreduce", red, Expected{size, iter}.sum(0));
      if (rank == root)
        for (int r = 0; r < size; r++)
          for (int k = 0; k <= r; k++) {
            char what[96];
            snprintf(what, sizeof what, "P7 Igatherv[%d][%d]", r, k);
            check(rank, iter, what, recvv[displs[r] + k], gatVal(r, iter) + 10L * k);
          }
    }

    /* Part 8 */
    if ((parts & 16) && iter % 10 == 0) {
      const int root = (iter / 10) % size;
      std::vector<long> sendv(rank + 1), allv, gat(size, -1);
      std::vector<int> counts(size), displs(size);
      int off = 0;
      for (int r = 0; r < size; r++) { counts[r] = r + 1; displs[r] = off; off += r + 1; }
      allv.assign(off, -1);
      for (int k = 0; k <= rank; k++) sendv[k] = gatVal(rank, iter) + 10L * k;
      const long inG = gatVal(rank, iter);
      red = -1;
      const int kind = (iter / 10) % 4;
      MPI_Request q;
      /* Every rank posts before AMPI_Migrate and waits after it: the valid
         collective order (see the note at the top). One collective per
         stage, rotating. */
      if (kind == 0) MPI_Iallreduce(&in, &red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD, &q);
      if (kind == 1) MPI_Iallgatherv(sendv.data(), rank + 1, MPI_LONG, allv.data(), counts.data(),
                                     displs.data(), MPI_LONG, MPI_COMM_WORLD, &q);
      if (kind == 2) MPI_Ibarrier(MPI_COMM_WORLD, &q);
      if (kind == 3) MPI_Igather(&inG, 1, MPI_LONG, gat.data(), 1, MPI_LONG, root, MPI_COMM_WORLD, &q);
      {
        int *pe, flag;
        MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &pe, &flag);
        const int peBefore = *pe;
        AMPI_Migrate(AMPI_INFO_LB_SYNC);
        MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &pe, &flag);
        if (*pe != peBefore) moves++;
      }
      MPI_Wait(&q, MPI_STATUS_IGNORE);
      if (kind == 0)
        check(rank, iter, "P8 Iallreduce", red, Expected{size, iter}.sum(0));
      if (kind == 1)
        for (int r = 0; r < size; r++)
          for (int k = 0; k <= r; k++) {
            char what[96];
            snprintf(what, sizeof what, "P8 Iallgatherv[%d][%d]", r, k);
            check(rank, iter, what, allv[displs[r] + k], gatVal(r, iter) + 10L * k);
          }
      if (kind == 3 && rank == root)
        for (int r = 0; r < size; r++) {
          char what[96];
          snprintf(what, sizeof what, "P8 Igather[%d]", r);
          check(rank, iter, what, gat[r], gatVal(r, iter));
        }
    }
  }

  int total = 0, totalMoves = 0;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&moves, &totalMoves, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (parts & 16)
      printf("collmatch: %d rank migrations across AMPI_Migrate\n", totalMoves);
    if (total == 0)
      printf("collmatch: all %d iterations passed on %d ranks\n", iters, size);
    else
      printf("collmatch: %d FAILED checks on %d ranks\n", total, size);
    fflush(stdout);
  }
  /* AMPI's MPI_Finalize never returns (it ends in AMPI_Exit(0)), so a
     failure must be reported through the exit status before it. */
  if (total != 0)
    MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Comm_free(&sub);
  MPI_Finalize();
  return 0;
}
