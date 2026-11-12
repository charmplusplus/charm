// The neighbor collectives, blocking and nonblocking, on Cartesian, graph and
// distributed-graph communicators. Each rank sends rank*1000+i as block i, so
// a received value says which rank sent it and which of its blocks it was.
// MPI 3.1 section 7.6 fixes the neighbor order: for a Cartesian topology the
// neighbor at -1 and then at +1 in each dimension, with an empty slot for
// MPI_PROC_NULL; for a graph the order of MPI_Graph_neighbors; for a
// distributed graph the sources and destinations as given. The Cartesian
// cases include a non-periodic grid (MPI_PROC_NULL slots), periodic
// dimensions of size 2 and 1 (the same rank at -1 and +1, so the blocks must
// match by order), and the test runs on 1 and 2 PEs (some neighbors local).
#include <cstdio>
#include <vector>
#include "mpi.h"

// AMPI ranks in one process share global variables, so the rank and the
// error count are passed around.
struct Ctx {
  int rank, size;
  long errs;
};

struct Topo {
  const char *name;
  MPI_Comm comm;
  std::vector<int> srcs;   // receive slot j comes from srcs[j] (or MPI_PROC_NULL)
  std::vector<int> fromIdx; // ... and is that rank's send block fromIdx[j]
  int nOut;                // number of send blocks
};

static void check(Ctx &c, const Topo &t, const char *call, const std::vector<int> &r, bool gather) {
  for (size_t j = 0; j < t.srcs.size(); j++) {
    int s = t.srcs[j];
    const int rank = c.rank;
    int want = (s == MPI_PROC_NULL) ? -1 : gather ? s * 1000 : s * 1000 + t.fromIdx[j];
    if (r[j] != want) {
      c.errs++;
      printf("rank %d: %s on %s: slot %zu is %d, expected %d\n", rank, call, t.name, j, r[j], want);
    }
  }
}

static void run(Ctx &c, const Topo &t) {
  const int rank = c.rank;
  const int nIn = t.srcs.size(), nOut = t.nOut;
  std::vector<int> s(nOut), r(nIn);
  for (int i = 0; i < nOut; i++) s[i] = rank * 1000 + i;
  // reversed layouts for the v and w variants
  std::vector<int> ones(nOut > nIn ? nOut : nIn, 1), sd(nOut), rd(nIn);
  std::vector<MPI_Aint> sdb(nOut), rdb(nIn);
  std::vector<MPI_Datatype> types(ones.size(), MPI_INT);
  std::vector<int> srev(nOut);
  for (int i = 0; i < nOut; i++) { sd[i] = nOut - 1 - i; sdb[i] = sd[i] * sizeof(int); srev[nOut - 1 - i] = s[i]; }
  for (int j = 0; j < nIn; j++) { rd[j] = nIn - 1 - j; rdb[j] = rd[j] * sizeof(int); }
  auto unrev = [&](std::vector<int> &v) { std::vector<int> u(nIn); for (int j = 0; j < nIn; j++) u[j] = v[nIn - 1 - j]; v = u; };
  const int me = rank * 1000;

  for (int nb = 0; nb < 2; nb++) {
    MPI_Request req;
    auto done = [&] { if (nb) MPI_Wait(&req, MPI_STATUS_IGNORE); };

    r.assign(nIn, -1);
    if (nb) MPI_Ineighbor_allgather(&me, 1, MPI_INT, r.data(), 1, MPI_INT, t.comm, &req);
    else MPI_Neighbor_allgather(&me, 1, MPI_INT, r.data(), 1, MPI_INT, t.comm);
    done();
    check(c, t, nb ? "MPI_Ineighbor_allgather" : "MPI_Neighbor_allgather", r, true);

    r.assign(nIn, -1);
    if (nb) MPI_Ineighbor_allgatherv(&me, 1, MPI_INT, r.data(), ones.data(), rd.data(), MPI_INT, t.comm, &req);
    else MPI_Neighbor_allgatherv(&me, 1, MPI_INT, r.data(), ones.data(), rd.data(), MPI_INT, t.comm);
    done();
    unrev(r);
    check(c, t, nb ? "MPI_Ineighbor_allgatherv" : "MPI_Neighbor_allgatherv", r, true);

    r.assign(nIn, -1);
    if (nb) MPI_Ineighbor_alltoall(s.data(), 1, MPI_INT, r.data(), 1, MPI_INT, t.comm, &req);
    else MPI_Neighbor_alltoall(s.data(), 1, MPI_INT, r.data(), 1, MPI_INT, t.comm);
    done();
    check(c, t, nb ? "MPI_Ineighbor_alltoall" : "MPI_Neighbor_alltoall", r, false);

    r.assign(nIn, -1);
    if (nb) MPI_Ineighbor_alltoallv(srev.data(), ones.data(), sd.data(), MPI_INT, r.data(), ones.data(), rd.data(), MPI_INT, t.comm, &req);
    else MPI_Neighbor_alltoallv(srev.data(), ones.data(), sd.data(), MPI_INT, r.data(), ones.data(), rd.data(), MPI_INT, t.comm);
    done();
    unrev(r);
    check(c, t, nb ? "MPI_Ineighbor_alltoallv" : "MPI_Neighbor_alltoallv", r, false);

    r.assign(nIn, -1);
    if (nb) MPI_Ineighbor_alltoallw(srev.data(), ones.data(), sdb.data(), types.data(), r.data(), ones.data(), rdb.data(), types.data(), t.comm, &req);
    else MPI_Neighbor_alltoallw(srev.data(), ones.data(), sdb.data(), types.data(), r.data(), ones.data(), rdb.data(), types.data(), t.comm);
    done();
    unrev(r);
    check(c, t, nb ? "MPI_Ineighbor_alltoallw" : "MPI_Neighbor_alltoallw", r, false);
  }
}

// A Cartesian communicator: slot 2d comes from the -1 neighbor's block for
// +1 (2d+1), slot 2d+1 from the +1 neighbor's block for -1 (2d).
static void cart(Ctx &c, const char *name, int ndims, const int *dims, const int *periods) {
  Topo t{name, MPI_COMM_NULL, {}, {}, 2 * ndims};
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &t.comm);
  if (t.comm == MPI_COMM_NULL) return;
  for (int d = 0; d < ndims; d++) {
    int lo, hi;
    MPI_Cart_shift(t.comm, d, 1, &lo, &hi);
    t.srcs.push_back(lo);
    t.fromIdx.push_back(2 * d + 1);
    t.srcs.push_back(hi);
    t.fromIdx.push_back(2 * d);
  }
  run(c, t);
  MPI_Comm_free(&t.comm);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Ctx c{rank, size, 0};
  const int half = size / 2;

  { int dims[1] = {size}, periods[1] = {1}; cart(c, "periodic ring", 1, dims, periods); }
  if (half > 0) {
    { int dims[2] = {2, half}, periods[2] = {0, 0}; cart(c, "2-D grid", 2, dims, periods); }
    { int dims[2] = {2, half}, periods[2] = {1, 1}; cart(c, "2-D torus", 2, dims, periods); }
  }
  { int dims[2] = {1, size}, periods[2] = {1, 0}; cart(c, "size-1 periodic dimension", 2, dims, periods); }

  // a graph ring in which rank r lists its edges as {r+1, r-1}; with 2 ranks
  // both edges join the same pair, and the k-th edge from r to x matches the
  // k-th from x to r
  {
    std::vector<int> index(size), edges;
    for (int r = 0; r < size; r++) {
      edges.push_back((r + 1) % size);
      edges.push_back((r + size - 1) % size);
      index[r] = edges.size();
    }
    Topo t{"graph ring", MPI_COMM_NULL, {}, {}, 0};
    MPI_Graph_create(MPI_COMM_WORLD, size, index.data(), edges.data(), 0, &t.comm);
    int n;
    MPI_Graph_neighbors_count(t.comm, rank, &n);
    t.srcs.resize(n);
    MPI_Graph_neighbors(t.comm, rank, n, t.srcs.data());
    t.nOut = n;
    for (int j = 0; j < n; j++) {
      int x = t.srcs[j], k = 0;
      for (int jj = 0; jj < j; jj++) k += (t.srcs[jj] == x);
      int i = 0;
      for (int e = 2 * x; e < 2 * x + 2; e++)
        if (edges[e] == rank && k-- == 0) i = e - 2 * x;
      t.fromIdx.push_back(i);
    }
    run(c, t);
    MPI_Comm_free(&t.comm);
  }

  // a distributed graph whose sources and destinations differ: rank r sends
  // to {r+1, r+2} and so receives from {r-2 (its block 1), r-1 (its block 0)}
  if (size >= 3) {
    int dsts[2] = {(rank + 1) % size, (rank + 2) % size};
    int srcs[2] = {(rank + size - 2) % size, (rank + size - 1) % size};
    Topo t{"distributed graph", MPI_COMM_NULL, {srcs[0], srcs[1]}, {1, 0}, 2};
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 2, srcs, MPI_UNWEIGHTED, 2, dsts,
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &t.comm);
    run(c, t);
    MPI_Comm_free(&t.comm);
  }

  long total = 0;
  MPI_Reduce(&c.errs, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("neighbor: %d ranks, errors=%ld: %s\n", size, total, total == 0 ? "PASSED" : "FAILED");
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
