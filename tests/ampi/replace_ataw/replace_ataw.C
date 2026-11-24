// MPI_Sendrecv_replace with a strided datatype, and MPI_Ialltoall, MPI_Ialltoallv
// and MPI_Ialltoallw (with and without MPI_IN_PLACE) completed by MPI_Wait.
// Each is run with a small and a large message, so that both the eager and
// the large-message paths are reached.
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mpi.h"

static int errors = 0;

static void check(bool ok, const char *what, int rank, int i) {
  if (!ok && errors++ < 10) printf("[%d] %s: wrong value at %d\n", rank, what, i);
}

static int value(int from, int to, int i) { return 1000000 * from + 1000 * to + i % 1000; }

// Ring exchange of every other int of a buffer of 2*n ints; the gaps must stay untouched.
static void testReplace(int rank, int size, int n) {
  MPI_Datatype strided;
  MPI_Type_vector(n, 1, 2, MPI_INT, &strided);
  MPI_Type_commit(&strided);
  std::vector<int> buf(2 * n);
  for (int i = 0; i < n; i++) { buf[2 * i] = value(rank, 0, i); buf[2 * i + 1] = -1; }
  int right = (rank + 1) % size, left = (rank + size - 1) % size;
  MPI_Status st;
  MPI_Sendrecv_replace(buf.data(), 1, strided, right, 7, left, 7, MPI_COMM_WORLD, &st);
  for (int i = 0; i < n; i++) {
    check(buf[2 * i] == value(left, 0, i), "Sendrecv_replace data", rank, 2 * i);
    check(buf[2 * i + 1] == -1, "Sendrecv_replace gap", rank, 2 * i + 1);
  }
  check(st.MPI_SOURCE == left, "Sendrecv_replace status", rank, 0);
  MPI_Type_free(&strided);
}

// Rank r sends n ints to each rank d, with byte displacements.
static void testIalltoallw(int rank, int size, int n, bool inPlace) {
  std::vector<int> counts(size, n), displs(size);
  std::vector<MPI_Datatype> types(size, MPI_INT);
  for (int d = 0; d < size; d++) displs[d] = d * n * (int)sizeof(int);
  std::vector<int> sbuf(size * n), rbuf(size * n, -1);
  std::vector<int> &out = inPlace ? rbuf : sbuf;
  for (int d = 0; d < size; d++)
    for (int i = 0; i < n; i++) out[d * n + i] = value(rank, d, i);
  MPI_Request req;
  MPI_Ialltoallw(inPlace ? MPI_IN_PLACE : sbuf.data(), counts.data(), displs.data(), types.data(),
                 rbuf.data(), counts.data(), displs.data(), types.data(), MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  const char *what = inPlace ? "Ialltoallw in place" : "Ialltoallw";
  for (int s = 0; s < size; s++)
    for (int i = 0; i < n; i++) check(rbuf[s * n + i] == value(s, rank, i), what, rank, s * n + i);
}

// The same exchange through MPI_Ialltoall (kind 0) and MPI_Ialltoallv (kind 1).
static void testIalltoall(int rank, int size, int n, bool inPlace, int kind) {
  std::vector<int> counts(size, n), displs(size);
  for (int d = 0; d < size; d++) displs[d] = d * n;
  std::vector<int> sbuf(size * n), rbuf(size * n, -1);
  std::vector<int> &out = inPlace ? rbuf : sbuf;
  for (int d = 0; d < size; d++)
    for (int i = 0; i < n; i++) out[d * n + i] = value(rank, d, i);
  void *send = inPlace ? MPI_IN_PLACE : sbuf.data();
  MPI_Request req;
  if (kind == 0)
    MPI_Ialltoall(send, n, MPI_INT, rbuf.data(), n, MPI_INT, MPI_COMM_WORLD, &req);
  else
    MPI_Ialltoallv(send, counts.data(), displs.data(), MPI_INT,
                   rbuf.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  const char *what = kind == 0 ? (inPlace ? "Ialltoall in place" : "Ialltoall")
                               : (inPlace ? "Ialltoallv in place" : "Ialltoallv");
  for (int s = 0; s < size; s++)
    for (int i = 0; i < n; i++) check(rbuf[s * n + i] == value(s, rank, i), what, rank, s * n + i);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int n : {5, 70000}) {
    testReplace(rank, size, n);
    testIalltoallw(rank, size, n, false);
    testIalltoallw(rank, size, n, true);
    for (int kind = 0; kind < 2; kind++) {
      testIalltoall(rank, size, n, false, kind);
      testIalltoall(rank, size, n, true, kind);
    }
  }
  int total = 0;
  MPI_Reduce(&errors, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) printf("replace_ataw: %d ranks, errors=%d\n", size, total);
  MPI_Finalize();
  return total != 0;
}
