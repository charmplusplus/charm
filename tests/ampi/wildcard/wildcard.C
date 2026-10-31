// A user receive or probe with MPI_ANY_TAG must match only user tags
// (0..MPI_TAG_UB), never the internal tags that AMPI's collectives use on
// the same communicator. Each phase posts MPI_Irecv(MPI_ANY_SOURCE,
// MPI_ANY_TAG), runs a collective with that receive pending, then satisfies
// the receive with a real user send and checks both results. A wildcard that
// takes an internal message either hangs the collective or completes the
// user receive with an internal tag.
//
// Phase 7 checks that MPI_Probe/MPI_Iprobe with MPI_ANY_TAG skip a queued
// internal (scatter) message. Phase 8 checks MPI_Ineighbor_alltoall on its
// own (its sends and receives must use the same internal tag).
//
// Usage: ./wildcard [phase]   (no argument runs every phase)
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "mpi.h"

// AMPI ranks are threads in one process, so no mutable globals: every rank
// keeps its state in its own Ctx.
static const int WBUF = 256;
struct Ctx {
  int wrank;
  int errors;
  int wbuf[WBUF];
};

#define CHECK(cond, ...) do { if (!(cond)) { c.errors++; \
  printf("[%d] FAIL: ", c.wrank); printf(__VA_ARGS__); printf("\n"); fflush(stdout); } } while (0)

static void postWild(Ctx &c, MPI_Comm comm, MPI_Request *req)
{
  for (int i = 0; i < WBUF; i++) c.wbuf[i] = -1;
  MPI_Irecv(c.wbuf, WBUF, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, req);
}

// Satisfy the pending wildcard receive with a user message from the left
// neighbor and check that it, not an internal message, completed it.
static void finishWild(Ctx &c, MPI_Comm comm, int phase, MPI_Request *req)
{
  int r, n;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &n);
  int left = (r - 1 + n) % n, right = (r + 1) % n;
  int tag = 100 + phase;
  int v = 1000 * phase + r;
  MPI_Send(&v, 1, MPI_INT, right, tag, comm);
  MPI_Status st;
  MPI_Wait(req, &st);
  bool ok = (st.MPI_TAG == tag && st.MPI_SOURCE == left && c.wbuf[0] == 1000 * phase + left);
  CHECK(ok, "phase %d: wildcard recv got tag %d src %d val %d, expected tag %d src %d val %d",
        phase, st.MPI_TAG, st.MPI_SOURCE, c.wbuf[0], tag, left, 1000 * phase + left);
  if (!ok) { // drain the user message so it does not leak into later phases
    int tmp;
    MPI_Recv(&tmp, 1, MPI_INT, left, tag, comm, MPI_STATUS_IGNORE);
  }
}

static void ringCart(MPI_Comm *cart)
{
  int n;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  int dims[1] = { n }, periods[1] = { 1 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, cart);
}

// Each rank sends its rank to both ring neighbors; the received pair, sorted,
// must be the sorted pair of neighbor ranks (AMPI orders neighbors by
// locality, so only the multiset is checked).
static void checkNbor(Ctx &c, MPI_Comm cart, int phase, const int rbuf[2])
{
  int r, n;
  MPI_Comm_rank(cart, &r);
  MPI_Comm_size(cart, &n);
  int exp[2] = { (r - 1 + n) % n, (r + 1) % n };
  int got[2] = { rbuf[0], rbuf[1] };
  std::sort(exp, exp + 2);
  std::sort(got, got + 2);
  CHECK(got[0] == exp[0] && got[1] == exp[1], "phase %d: neighbor alltoall got {%d,%d}, expected {%d,%d}",
        phase, got[0], got[1], exp[0], exp[1]);
}

static void runPhase(Ctx &c, int phase)
{
  int r, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Request wreq, req;

  switch (phase) {
  case 1: { // MPI_Alltoall
    int *sbuf = new int[n], *rbuf = new int[n];
    for (int i = 0; i < n; i++) { sbuf[i] = r * 100 + i; rbuf[i] = -1; }
    postWild(c, MPI_COMM_WORLD, &wreq);
    MPI_Alltoall(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < n; i++)
      CHECK(rbuf[i] == i * 100 + r, "phase 1: alltoall rbuf[%d]=%d, expected %d", i, rbuf[i], i * 100 + r);
    finishWild(c, MPI_COMM_WORLD, phase, &wreq);
    delete [] sbuf; delete [] rbuf;
    break;
  }
  case 2: { // MPI_Scatter
    int *sbuf = new int[n];
    for (int i = 0; i < n; i++) sbuf[i] = 500 + i;
    int v = -1;
    postWild(c, MPI_COMM_WORLD, &wreq);
    MPI_Scatter(sbuf, 1, MPI_INT, &v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CHECK(v == 500 + r, "phase 2: scatter got %d, expected %d", v, 500 + r);
    finishWild(c, MPI_COMM_WORLD, phase, &wreq);
    delete [] sbuf;
    break;
  }
  case 3: { // MPI_Scan
    int v = r + 1, s = -1;
    postWild(c, MPI_COMM_WORLD, &wreq);
    MPI_Scan(&v, &s, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    CHECK(s == (r + 1) * (r + 2) / 2, "phase 3: scan got %d, expected %d", s, (r + 1) * (r + 2) / 2);
    finishWild(c, MPI_COMM_WORLD, phase, &wreq);
    break;
  }
  case 4: { // MPI_Ibarrier + MPI_Wait
    postWild(c, MPI_COMM_WORLD, &wreq);
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    finishWild(c, MPI_COMM_WORLD, phase, &wreq);
    break;
  }
  case 5: { // MPI_Iallreduce + MPI_Wait
    int v = r, s = -1;
    postWild(c, MPI_COMM_WORLD, &wreq);
    MPI_Iallreduce(&v, &s, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    CHECK(s == n * (n - 1) / 2, "phase 5: iallreduce got %d, expected %d", s, n * (n - 1) / 2);
    finishWild(c, MPI_COMM_WORLD, phase, &wreq);
    break;
  }
  case 6: { // MPI_Neighbor_alltoall on a Cart communicator
    MPI_Comm cart;
    ringCart(&cart);
    int cr;
    MPI_Comm_rank(cart, &cr);
    int sbuf[2] = { cr, cr }, rbuf[2] = { -1, -1 };
    postWild(c, cart, &wreq);
    MPI_Neighbor_alltoall(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, cart);
    checkNbor(c, cart, phase, rbuf);
    finishWild(c, cart, phase, &wreq);
    MPI_Comm_free(&cart);
    break;
  }
  case 7: { // MPI_Probe / MPI_Iprobe with MPI_ANY_TAG behind a queued scatter message
    int *sbuf = new int[n];
    for (int i = 0; i < n; i++) sbuf[i] = 800 + i;
    int v = -1;
    if (r == 0) {
      MPI_Scatter(sbuf, 1, MPI_INT, &v, 1, MPI_INT, 0, MPI_COMM_WORLD);
      for (int i = 1; i < n; i++) {
        int u = 900 + i;
        MPI_Send(&u, 1, MPI_INT, i, 9, MPI_COMM_WORLD);
      }
    } else {
      // Rank 0 sent its scatter block before the tag-9 message, so the
      // scatter message is already queued here when the probe sees tag 9.
      MPI_Status st;
      int flag = 0;
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      CHECK(st.MPI_TAG == 9, "phase 7: MPI_Probe(0, ANY_TAG) returned tag %d, expected 9", st.MPI_TAG);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
      CHECK(flag && st.MPI_TAG == 9, "phase 7: MPI_Iprobe(ANY, ANY) returned flag %d tag %d, expected 1, 9",
            flag, st.MPI_TAG);
      int u = -1;
      MPI_Recv(&u, 1, MPI_INT, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      CHECK(u == 900 + r, "phase 7: user recv got %d, expected %d", u, 900 + r);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
      CHECK(!flag, "phase 7: MPI_Iprobe(ANY, ANY) found a message with tag %d; only an internal one is queued",
            st.MPI_TAG);
      MPI_Scatter(sbuf, 1, MPI_INT, &v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    CHECK(v == 800 + r, "phase 7: scatter got %d, expected %d", v, 800 + r);
    delete [] sbuf;
    break;
  }
  case 8: { // MPI_Ineighbor_alltoall + MPI_Wait on a Cart communicator
    MPI_Comm cart;
    ringCart(&cart);
    int cr;
    MPI_Comm_rank(cart, &cr);
    int sbuf[2] = { cr, cr }, rbuf[2] = { -1, -1 };
    MPI_Ineighbor_alltoall(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, cart, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    checkNbor(c, cart, phase, rbuf);
    MPI_Comm_free(&cart);
    break;
  }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  Ctx *ctx = new Ctx();
  Ctx &c = *ctx;
  c.errors = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &c.wrank);
  int wrank = c.wrank;
  int n;
  MPI_Comm_size(MPI_COMM_WORLD, &n);

  const int nphases = 8;
  int only = (argc > 1) ? atoi(argv[1]) : 0;
  if (n < 2) {
    if (wrank == 0) printf("wildcard: needs at least 2 ranks, skipping\n");
  } else {
    for (int p = 1; p <= nphases; p++) {
      if (only && p != only) continue;
      runPhase(c, p);
      if (wrank == 0) { printf("wildcard: phase %d done\n", p); fflush(stdout); }
    }
  }

  int total = 0;
  MPI_Allreduce(&c.errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (wrank == 0) printf("wildcard: %s (%d errors)\n", total ? "FAILED" : "passed", total);
  delete ctx;
  if (total != 0) MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
  return 0;
}
