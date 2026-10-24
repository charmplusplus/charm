/* MPI_Scan, MPI_Exscan, MPI_Iscan and MPI_Iexscan, checked against a
   prefix computed serially on every rank.

   Each rank's input is a deterministic function of (rank, element), so every
   rank can compute the inclusive or exclusive prefix over ranks 0..r on its
   own. Covered:
     - int and double, MPI_SUM and MPI_MAX, counts 1, 4 and 20000 (160 KB of
       doubles per round message);
     - a non-commutative user op (composition of affine maps mod a prime),
       which checks that partial results are combined in rank order;
     - MPI_IN_PLACE;
     - a communicator from MPI_Comm_split, so communicator ranks differ from
       world ranks;
     - MPI_Iscan / MPI_Iexscan completed by MPI_Wait, by an MPI_Test loop, and
       by one MPI_Waitall over several outstanding scans;
     - progress: MPI_Iscan must return without waiting for the other ranks
       (rank 0 sends to rank 1 after its MPI_Iscan, and rank 1 enters the scan
       only after receiving it), and a scan must advance while a rank is
       blocked in MPI_Recv on something else (rank 0 waits for a message the
       last rank sends only after its own scan has completed);
     - MPI_Op_free on a user op while scans using it are outstanding;
     - with +balancer, an AMPI_Migrate while scans are outstanding.

   MPI_Exscan's recvbuf on rank 0 is undefined, so it is not checked there.
   The rank count that exposed the MPI_Exscan hang is 3 (any count that is not
   a power of two), so the Makefile runs +vp 3, 5 and 7 as well as 4.

   AMPI's MPI_Finalize never returns, so a failure is reported through
   MPI_Abort, which makes the run exit nonzero. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* No mutable globals: AMPI ranks are threads in one process and would share
   them. Per-rank state lives in a Ctx passed down. */
struct Ctx {
  int worldRank;
  int failures;
  MPI_Op affineOp;
};

static const long long MODP = 1000003;

/* Non-commutative op on pairs (a, b), each pair the map x -> a*x + b mod P.
   inout = in followed by inout, i.e. the lower ranks' map is applied first. */
static void affineCompose(void *invec, void *inoutvec, int *len, MPI_Datatype *dt)
{
  const int *in = (const int *)invec;
  int *io = (int *)inoutvec;
  for (int i = 0; i + 1 < *len; i += 2) {
    long long a1 = in[i], b1 = in[i+1], a2 = io[i], b2 = io[i+1];
    io[i]   = (int)((a1 * a2) % MODP);
    io[i+1] = (int)((a2 * b1 + b2) % MODP);
  }
}

// A different function for the op created after the free: if a pending scan
// looked its op up again, it would find this one and get wrong values.
static void otherOp(void *invec, void *inoutvec, int *len, MPI_Datatype *dt)
{
  for (int i = 0; i < *len; i++) ((int *)inoutvec)[i] += ((const int *)invec)[i] + 1;
}

static int ival(int rank, int i) { return ((rank * 7 + i * 3) % 11) - 5; }
static double dval(int rank, int i) { return 0.5 * (((rank * 5 + i * 13) % 17) - 8); }
static void affval(int rank, int i, int *pair)
{
  pair[0] = (rank * 31 + i * 7) % 1000 + 2;
  pair[1] = (rank * 17 + i * 11) % 1000 + 1;
}

enum Kind { K_INT_SUM, K_INT_MAX, K_DBL_SUM, K_DBL_MAX, K_AFFINE, K_COUNT };
static const char *kindName[] = {"int SUM", "int MAX", "double SUM", "double MAX", "affine (non-commutative)"};

struct Case {
  Kind kind;
  int count;          // elements (for K_AFFINE: ints, i.e. 2 per map)
  bool exclusive;
  std::vector<char> send, recv, want;
  MPI_Datatype type;
  MPI_Op op;
};

static void fill(Case &c, int rank, const Ctx &x)
{
  bool isDouble = (c.kind == K_DBL_SUM || c.kind == K_DBL_MAX);
  size_t esz = isDouble ? sizeof(double) : sizeof(int);
  c.send.assign(c.count * esz, 0);
  c.recv.assign(c.count * esz, 0);
  c.want.assign(c.count * esz, 0);
  c.type = isDouble ? MPI_DOUBLE : MPI_INT;
  c.op = (c.kind == K_INT_SUM || c.kind == K_DBL_SUM) ? MPI_SUM
       : (c.kind == K_AFFINE) ? x.affineOp : MPI_MAX;

  int last = c.exclusive ? rank - 1 : rank;
  if (c.kind == K_AFFINE) {
    int *s = (int *)c.send.data(), *w = (int *)c.want.data();
    for (int i = 0; i + 1 < c.count; i += 2) {
      affval(rank, i, &s[i]);
      long long a = 1, b = 0;              // identity map
      for (int r = 0; r <= last; r++) {    // apply rank 0's map first
        int p[2]; affval(r, i, p);
        b = (p[0] * b + p[1]) % MODP;
        a = (a * p[0]) % MODP;
      }
      w[i] = (int)a; w[i+1] = (int)b;
    }
    for (size_t i = 0; i < c.count * esz; i++) c.recv[i] = (char)0xA5;
    return;
  }
  for (int i = 0; i < c.count; i++) {
    if (isDouble) {
      ((double *)c.send.data())[i] = dval(rank, i);
      double acc = (c.kind == K_DBL_SUM) ? 0.0 : -1e300;
      for (int r = 0; r <= last; r++)
        acc = (c.kind == K_DBL_SUM) ? acc + dval(r, i) : (dval(r, i) > acc ? dval(r, i) : acc);
      ((double *)c.want.data())[i] = acc;
      ((double *)c.recv.data())[i] = -12345.0;
    } else {
      ((int *)c.send.data())[i] = ival(rank, i);
      int acc = (c.kind == K_INT_SUM) ? 0 : -1000000;
      for (int r = 0; r <= last; r++)
        acc = (c.kind == K_INT_SUM) ? acc + ival(r, i) : (ival(r, i) > acc ? ival(r, i) : acc);
      ((int *)c.want.data())[i] = acc;
      ((int *)c.recv.data())[i] = -12345;
    }
  }
}

static void verify(const Case &c, int rank, const char *how, Ctx &x)
{
  if (c.exclusive && rank == 0) return;   // undefined on rank 0
  bool isDouble = (c.kind == K_DBL_SUM || c.kind == K_DBL_MAX);
  for (int i = 0; i < c.count; i++) {
    bool ok;
    double got, want;
    if (isDouble) {
      got = ((const double *)c.recv.data())[i];
      want = ((const double *)c.want.data())[i];
      ok = (got == want);   // sums of halves are exact
    } else {
      got = ((const int *)c.recv.data())[i];
      want = ((const int *)c.want.data())[i];
      ok = (got == want);
    }
    if (!ok) {
      printf("FAIL: world rank %d (comm rank %d): %s %s count %d element %d = %g, expected %g\n",
             x.worldRank, rank, how, kindName[c.kind], c.count, i, got, want);
      fflush(stdout);
      x.failures++;
      return;
    }
  }
}

static void runSuite(MPI_Comm comm, const char *commName, Ctx &x)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const int counts[] = {1, 4, 20000};
  char how[128];

  for (int ex = 0; ex <= 1; ex++) {
    for (int k = 0; k < K_COUNT; k++) {
      for (int ci = 0; ci < 3; ci++) {
        Case c;
        c.kind = (Kind)k;
        c.count = (k == K_AFFINE) ? 2 * counts[ci] : counts[ci];
        c.exclusive = ex;

        // Blocking
        fill(c, rank, x);
        if (ex) MPI_Exscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm);
        else    MPI_Scan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm);
        snprintf(how, sizeof how, "%s %s", commName, ex ? "MPI_Exscan" : "MPI_Scan");
        verify(c, rank, how, x);

        // Blocking, MPI_IN_PLACE
        fill(c, rank, x);
        memcpy(c.recv.data(), c.send.data(), c.send.size());
        if (ex) MPI_Exscan(MPI_IN_PLACE, c.recv.data(), c.count, c.type, c.op, comm);
        else    MPI_Scan(MPI_IN_PLACE, c.recv.data(), c.count, c.type, c.op, comm);
        snprintf(how, sizeof how, "%s %s IN_PLACE", commName, ex ? "MPI_Exscan" : "MPI_Scan");
        verify(c, rank, how, x);

        // Nonblocking + MPI_Wait
        fill(c, rank, x);
        MPI_Request req;
        if (ex) MPI_Iexscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
        else    MPI_Iscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        snprintf(how, sizeof how, "%s %s+Wait", commName, ex ? "MPI_Iexscan" : "MPI_Iscan");
        verify(c, rank, how, x);
        if (req != MPI_REQUEST_NULL) {
          printf("FAIL: world rank %d: request not MPI_REQUEST_NULL after MPI_Wait\n", x.worldRank);
          x.failures++;
        }

        // Nonblocking + MPI_Test loop
        fill(c, rank, x);
        if (ex) MPI_Iexscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
        else    MPI_Iscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
        int flag = 0;
        while (!flag) MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        snprintf(how, sizeof how, "%s %s+Test", commName, ex ? "MPI_Iexscan" : "MPI_Iscan");
        verify(c, rank, how, x);
      }
    }
  }

  // Several outstanding nonblocking scans, completed by one MPI_Waitall.
  // Also mixes in a blocking scan while they are outstanding.
  for (int iter = 0; iter < 3; iter++) {
    std::vector<Case> cs(2 * K_COUNT);
    std::vector<MPI_Request> reqs(cs.size());
    for (size_t j = 0; j < cs.size(); j++) {
      Case &c = cs[j];
      c.kind = (Kind)(j % K_COUNT);
      c.exclusive = (j >= K_COUNT);
      int n = (iter == 2) ? 5000 : 3;
      c.count = (c.kind == K_AFFINE) ? 2 * n : n;
      fill(c, rank, x);
      if (c.exclusive) MPI_Iexscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &reqs[j]);
      else             MPI_Iscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &reqs[j]);
    }
    Case mid;
    mid.kind = K_AFFINE; mid.count = 2; mid.exclusive = (iter == 1);
    fill(mid, rank, x);
    if (mid.exclusive) MPI_Exscan(mid.send.data(), mid.recv.data(), mid.count, mid.type, mid.op, comm);
    else               MPI_Scan(mid.send.data(), mid.recv.data(), mid.count, mid.type, mid.op, comm);
    snprintf(how, sizeof how, "%s blocking scan between I-scans (iter %d)", commName, iter);
    verify(mid, rank, how, x);

    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    for (size_t j = 0; j < cs.size(); j++) {
      snprintf(how, sizeof how, "%s Waitall %s (iter %d)", commName,
               cs[j].exclusive ? "MPI_Iexscan" : "MPI_Iscan", iter);
      verify(cs[j], rank, how, x);
    }
  }
}

static void progressTests(MPI_Comm comm, Ctx &x)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size < 2) return;
  char how[128];

  for (int ex = 0; ex <= 1; ex++) {
    // (a) MPI_Iscan returns before the other ranks have entered the scan
    Case c;
    c.kind = K_AFFINE; c.count = 2; c.exclusive = ex;
    fill(c, rank, x);
    MPI_Request req;
    int token = 0;
    if (rank == 1) MPI_Recv(&token, 1, MPI_INT, 0, 77, comm, MPI_STATUS_IGNORE);
    if (ex) MPI_Iexscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
    else    MPI_Iscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
    if (rank == 0) MPI_Send(&token, 1, MPI_INT, 1, 77, comm);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    snprintf(how, sizeof how, "%s returns before peers enter", ex ? "MPI_Iexscan" : "MPI_Iscan");
    verify(c, rank, how, x);

    // (b) the scan advances while rank 0 is blocked in MPI_Recv
    fill(c, rank, x);
    if (ex) MPI_Iexscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
    else    MPI_Iscan(c.send.data(), c.recv.data(), c.count, c.type, c.op, comm, &req);
    if (rank == 0) {
      MPI_Recv(&token, 1, MPI_INT, size - 1, 78, comm, MPI_STATUS_IGNORE);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (rank == size - 1) MPI_Send(&token, 1, MPI_INT, 0, 78, comm);
    }
    snprintf(how, sizeof how, "%s progresses during MPI_Recv", ex ? "MPI_Iexscan" : "MPI_Iscan");
    verify(c, rank, how, x);
  }
}

// MPI_Op_free on a user op while scans using it are outstanding: the pending
// scans must complete normally (MPI-3.1 sec. 5.9.5). Rank 0 frees the op and
// creates another one (which may reuse the freed slot) before the other ranks
// have entered the scans, so rounds run after the free.
static void opFreeTest(MPI_Comm comm, Ctx &x)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  Ctx y = x;
  MPI_Op_create(affineCompose, 0, &y.affineOp);
  Case cs[2];
  MPI_Request reqs[2];
  for (int j = 0; j < 2; j++) {
    cs[j].kind = K_AFFINE; cs[j].count = 6; cs[j].exclusive = j;
    fill(cs[j], rank, y);
  }
  int token = 0;
  if (rank != 0 && size > 1) MPI_Recv(&token, 1, MPI_INT, 0, 79, comm, MPI_STATUS_IGNORE);
  MPI_Iscan(cs[0].send.data(), cs[0].recv.data(), cs[0].count, MPI_INT, y.affineOp, comm, &reqs[0]);
  MPI_Iexscan(cs[1].send.data(), cs[1].recv.data(), cs[1].count, MPI_INT, y.affineOp, comm, &reqs[1]);
  MPI_Op_free(&y.affineOp);
  MPI_Op other;
  MPI_Op_create(otherOp, 1, &other);   // may take the freed slot
  if (rank == 0) for (int r = 1; r < size; r++) MPI_Send(&token, 1, MPI_INT, r, 79, comm);
  MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
  MPI_Op_free(&other);
  x.failures = y.failures;
  verify(cs[0], rank, "MPI_Iscan after MPI_Op_free", x);
  verify(cs[1], rank, "MPI_Iexscan after MPI_Op_free", x);
}

#ifdef AMPI
// Migrate while scans are outstanding (with +balancer, ranks move and their
// posted scan requests are packed and unpacked).
static void migrateTest(MPI_Comm comm, Ctx &x)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  std::vector<Case> cs(2);
  MPI_Request reqs[2];
  for (int j = 0; j < 2; j++) {
    cs[j].kind = K_AFFINE; cs[j].count = 4; cs[j].exclusive = j;
    fill(cs[j], rank, x);
  }
  // Rank 0 starts its scans only after AMPI_Migrate, so every other rank's
  // scans are still waiting for rank 0's contribution while it migrates.
  if (rank != 0) {
    MPI_Iscan(cs[0].send.data(), cs[0].recv.data(), 4, MPI_INT, x.affineOp, comm, &reqs[0]);
    MPI_Iexscan(cs[1].send.data(), cs[1].recv.data(), 4, MPI_INT, x.affineOp, comm, &reqs[1]);
  }
  int *pe, flag, before, after;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &pe, &flag);
  before = *pe;
  AMPI_Migrate(AMPI_INFO_LB_SYNC);
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &pe, &flag);
  after = *pe;
  if (rank == 0) {
    MPI_Iscan(cs[0].send.data(), cs[0].recv.data(), 4, MPI_INT, x.affineOp, comm, &reqs[0]);
    MPI_Iexscan(cs[1].send.data(), cs[1].recv.data(), 4, MPI_INT, x.affineOp, comm, &reqs[1]);
  }
  MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
  int moved = (before != after), totalMoved = 0;
  MPI_Reduce(&moved, &totalMoved, 1, MPI_INT, MPI_SUM, 0, comm);
  if (rank == 0) printf("scan: %d ranks changed PE in AMPI_Migrate with scans outstanding\n", totalMoved);
  verify(cs[0], rank, "MPI_Iscan across AMPI_Migrate", x);
  verify(cs[1], rank, "MPI_Iexscan across AMPI_Migrate", x);
}
#endif

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int size;
  Ctx x;
  x.failures = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &x.worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Op_create(affineCompose, 0, &x.affineOp);

  runSuite(MPI_COMM_WORLD, "WORLD", x);
  progressTests(MPI_COMM_WORLD, x);
  opFreeTest(MPI_COMM_WORLD, x);
#ifdef AMPI
  migrateTest(MPI_COMM_WORLD, x);
#endif

  // Reverse-ordered split: communicator rank differs from world rank.
  MPI_Comm sub;
  MPI_Comm_split(MPI_COMM_WORLD, x.worldRank % 2, size - x.worldRank, &sub);
  runSuite(sub, "split", x);
  MPI_Comm_free(&sub);

  MPI_Op_free(&x.affineOp);

  int total = 0;
  MPI_Allreduce(&x.failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (x.worldRank == 0) {
    if (total == 0) printf("scan: all tests passed on %d ranks\n", size);
    else            printf("scan: %d failures on %d ranks\n", total, size);
    fflush(stdout);
  }
  if (total != 0) MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
  return 0;
}
