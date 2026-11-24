// campi test: the neighbor collectives on a periodic 1-D Cartesian ring
// (neighbors left, then right) and on a directed distributed-graph ring (in
// from left, out to right); MPI_Ireduce_scatter_block, MPI_Ireduce_scatter,
// MPI_Comm_idup and MPI_Comm_idup_with_info, which AMPI completes before they
// return.
#include <cstdio>
#include <vector>
#include "mpi.h"
#include "campi.h"

CAMPI_MAIN(argc, argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int right = (rank + 1) % size, left = (rank + size - 1) % size;
  long myerr = 0;
#define C(c)                                                                  \
  do {                                                                        \
    if (!(c)) { myerr++; printf("rank %d: check failed at line %d: %s\n",     \
                                rank, __LINE__, #c); }                        \
  } while (0)

  MPI_Comm ring;
  {
    int dims[1] = {size}, periods[1] = {1};
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &ring);
  }

  // 1. MPI_Neighbor_allgather
  {
    int r[2] = {-1, -1};
    MPI_Neighbor_allgather(&rank, 1, MPI_INT, r, 1, MPI_INT, ring);
    C(r[0] == left && r[1] == right);
  }

  // 2. MPI_Neighbor_allgatherv: the blocks land in reverse order
  {
    int r[2] = {-1, -1}, counts[2] = {1, 1}, displs[2] = {1, 0};
    MPI_Neighbor_allgatherv(&rank, 1, MPI_INT, r, counts, displs, MPI_INT, ring);
    C(r[1] == left && r[0] == right);
  }

  // 3. MPI_Neighbor_alltoall: block 0 goes left, block 1 right
  {
    int s[2] = {10 * rank, 10 * rank + 1}, r[2] = {-1, -1};
    MPI_Neighbor_alltoall(s, 1, MPI_INT, r, 1, MPI_INT, ring);
    C(r[0] == 10 * left + 1 && r[1] == 10 * right);
  }

  // 4. MPI_Neighbor_alltoallv: two ints left, one int right
  {
    int s[3] = {10 * rank, 10 * rank + 1, 10 * rank + 2}, r[3] = {-1, -1, -1};
    int scounts[2] = {2, 1}, sdispls[2] = {0, 2}, rcounts[2] = {1, 2}, rdispls[2] = {0, 1};
    MPI_Neighbor_alltoallv(s, scounts, sdispls, MPI_INT, r, rcounts, rdispls, MPI_INT, ring);
    C(r[0] == 10 * left + 2 && r[1] == 10 * right && r[2] == 10 * right + 1);
  }
  MPI_Comm_free(&ring);

  // 5. MPI_Neighbor_alltoallw on a directed ring: receive from left, send right
  {
    MPI_Comm g;
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 1, &left, MPI_UNWEIGHTED, 1, &right,
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &g);
    double s = 100.0 * rank, r = -1;
    int one = 1;
    MPI_Aint zero = 0;
    MPI_Datatype t = MPI_DOUBLE;
    MPI_Neighbor_alltoallw(&s, &one, &zero, &t, &r, &one, &zero, &t, g);
    C(r == 100.0 * left);
    MPI_Comm_free(&g);
  }

  // 6. MPI_Ireduce_scatter_block: rank i gets the sum over ranks of rank + i
  {
    std::vector<int> s(size);
    for (int i = 0; i < size; i++) s[i] = rank + i;
    int r = -1;
    MPI_Request req;
    MPI_Ireduce_scatter_block(s.data(), &r, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    C(r == size * (size - 1) / 2 + size * rank);
  }

  // 7. MPI_Ireduce_scatter: rank i gets i % 2 + 1 elements, element k of the
  // vector being the sum over ranks of rank + k
  {
    std::vector<int> counts(size);
    int total = 0, first = 0;
    for (int i = 0; i < size; i++) {
      counts[i] = i % 2 + 1;
      if (i < rank) first += counts[i];
      total += counts[i];
    }
    std::vector<int> s(total), r(counts[rank], -1);
    for (int k = 0; k < total; k++) s[k] = rank + k;
    MPI_Request req;
    MPI_Ireduce_scatter(s.data(), r.data(), counts.data(), MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                        &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    for (int k = 0; k < counts[rank]; k++)
      C(r[k] == size * (size - 1) / 2 + size * (first + k));
  }

  // 8. MPI_Comm_idup and MPI_Comm_idup_with_info
  {
    MPI_Comm d, e;
    MPI_Request reqs[2];
    MPI_Comm_idup(MPI_COMM_WORLD, &d, &reqs[0]);
    MPI_Comm_idup_with_info(d, MPI_INFO_NULL, &e, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    int sum = -1, dsize = -1, erank = -1;
    MPI_Comm_size(d, &dsize);
    MPI_Comm_rank(e, &erank);
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, e);
    C(dsize == size && erank == rank && sum == size * (size - 1) / 2);
    MPI_Comm_free(&e);
    MPI_Comm_free(&d);
  }

  long errs = 0;
  MPI_Reduce(&myerr, &errs, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("campi neighbor: %d ranks, errors=%ld: %s\n", size, errs, errs == 0 ? "PASSED" : "FAILED");
  MPI_Finalize();
  co_return errs ? 1 : 0;
}
