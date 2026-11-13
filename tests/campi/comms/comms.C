// campi test: the communicator constructors that wait for AMPI to create a
// new communicator: MPI_Comm_split_type, MPI_Cart_create (also with ranks
// left over), MPI_Cart_sub (also to 0 dimensions), MPI_Graph_create,
// MPI_Dist_graph_create_adjacent, MPI_Dist_graph_create,
// MPI_Intercomm_create, MPI_Intercomm_merge and MPI_Comm_create_group
// (called by the group's members only). Each new communicator is used for an
// MPI_Allreduce, which checks its membership.
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
  // the sum of the world ranks of comm's members, and its size
  auto check_members = [&](MPI_Comm comm, int expectSize, int expectSum, int line) -> campi::task<void> {
    int csize = -1, sum = -1;
    MPI_Comm_size(comm, &csize);
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, comm);
    if (csize != expectSize || sum != expectSum) {
      myerr++;
      printf("rank %d: line %d: size %d (expected %d), rank sum %d (expected %d)\n",
             rank, line, csize, expectSize, sum, expectSum);
    }
  };
  const int all = size * (size - 1) / 2;

  // 1. MPI_Comm_split_type: every rank of this test shares one host
  {
    MPI_Comm c;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &c);
    co_await check_members(c, size, all, __LINE__);
    MPI_Comm_free(&c);
  }

  // 2. MPI_Cart_create: a periodic 2 x (size/2) grid; with an odd size the
  // last rank is left out
  const int cols = size / 2, gridSize = 2 * cols;
  MPI_Comm cart;
  {
    int dims[2] = {2, cols}, periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    if (rank < gridSize) {
      C(cart != MPI_COMM_NULL);
      co_await check_members(cart, gridSize, gridSize * (gridSize - 1) / 2, __LINE__);
      int coords[2], src, dst;
      MPI_Cart_coords(cart, rank, 2, coords);
      C(coords[0] == rank / cols && coords[1] == rank % cols);
      MPI_Cart_shift(cart, 1, 1, &src, &dst);
      C(dst == coords[0] * cols + (coords[1] + 1) % cols);
    } else {
      C(cart == MPI_COMM_NULL);
    }
  }

  // 3. MPI_Cart_sub: the rows of the grid, then 0 dimensions
  if (cart != MPI_COMM_NULL) {
    int keep[2] = {0, 1};
    MPI_Comm row;
    MPI_Cart_sub(cart, keep, &row);
    const int r0 = (rank / cols) * cols;
    co_await check_members(row, cols, cols * r0 + cols * (cols - 1) / 2, __LINE__);
    int ndims = -1;
    MPI_Cartdim_get(row, &ndims);
    C(ndims == 1);
    MPI_Comm_free(&row);

    int none[2] = {0, 0};
    MPI_Comm point;
    MPI_Cart_sub(cart, none, &point);
    if (rank == 0) {
      C(point != MPI_COMM_NULL);
      MPI_Cartdim_get(point, &ndims);
      C(ndims == 0);
      MPI_Comm_free(&point);
    } else {
      C(point == MPI_COMM_NULL);
    }
    MPI_Comm_free(&cart);
  }

  // 4. MPI_Graph_create: a ring
  {
    std::vector<int> index(size), edges(2 * size);
    for (int i = 0; i < size; i++) {
      index[i] = 2 * (i + 1);
      edges[2 * i] = (i + size - 1) % size;
      edges[2 * i + 1] = (i + 1) % size;
    }
    MPI_Comm g;
    MPI_Graph_create(MPI_COMM_WORLD, size, index.data(), edges.data(), 0, &g);
    co_await check_members(g, size, all, __LINE__);
    int nn = -1, nb[2] = {-1, -1};
    MPI_Graph_neighbors_count(g, rank, &nn);
    MPI_Graph_neighbors(g, rank, 2, nb);
    C(nn == 2 && nb[0] == left && nb[1] == right);
    MPI_Comm_free(&g);
  }

  // 5. MPI_Dist_graph_create_adjacent: a directed ring (left -> rank -> right)
  {
    MPI_Comm g;
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 1, &left, MPI_UNWEIGHTED, 1, &right,
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &g);
    co_await check_members(g, size, all, __LINE__);
    int in = -1, out = -1, weighted = -1, src = -1, dst = -1;
    MPI_Dist_graph_neighbors_count(g, &in, &out, &weighted);
    MPI_Dist_graph_neighbors(g, 1, &src, MPI_UNWEIGHTED, 1, &dst, MPI_UNWEIGHTED);
    C(in == 1 && out == 1 && src == left && dst == right);
    MPI_Comm_free(&g);
  }

  // 6. MPI_Dist_graph_create: the same ring, each rank giving its own out-edge
  {
    MPI_Comm g;
    int one = 1;
    MPI_Dist_graph_create(MPI_COMM_WORLD, 1, &rank, &one, &right, MPI_UNWEIGHTED,
                          MPI_INFO_NULL, 0, &g);
    co_await check_members(g, size, all, __LINE__);
    int in = -1, out = -1, weighted = -1, src = -1, dst = -1;
    MPI_Dist_graph_neighbors_count(g, &in, &out, &weighted);
    MPI_Dist_graph_neighbors(g, 1, &src, MPI_UNWEIGHTED, 1, &dst, MPI_UNWEIGHTED);
    C(in == 1 && out == 1 && src == left && dst == right);
    MPI_Comm_free(&g);
  }

  // 7. MPI_Intercomm_create between the even and the odd ranks (leaders: world
  // ranks 0 and 1), then MPI_Intercomm_merge with the odd group high
  {
    MPI_Comm half, inter, merged;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
    MPI_Intercomm_create(half, 0, MPI_COMM_WORLD, 1 - rank % 2, 7, &inter);
    int flag = 0, rsize = -1;
    MPI_Comm_test_inter(inter, &flag);
    MPI_Comm_remote_size(inter, &rsize);
    const int evens = (size + 1) / 2;
    C(flag && rsize == (rank % 2 ? evens : size - evens));
    MPI_Intercomm_merge(inter, rank % 2, &merged);
    co_await check_members(merged, size, all, __LINE__);
    int mrank = -1;
    MPI_Comm_rank(merged, &mrank);
    C(mrank == (rank % 2 ? evens + rank / 2 : rank / 2));
    MPI_Comm_free(&merged);
    MPI_Comm_free(&inter);
    MPI_Comm_free(&half);
  }

  // 8. MPI_Comm_create_group: the even ranks, in reverse order; the odd ranks
  // do not call it
  {
    MPI_Group world, evens;
    MPI_Comm_group(MPI_COMM_WORLD, &world);
    std::vector<int> members;
    for (int i = (size - 1) / 2 * 2; i >= 0; i -= 2) members.push_back(i);
    MPI_Group_incl(world, members.size(), members.data(), &evens);
    if (rank % 2 == 0) {
      MPI_Comm c;
      MPI_Comm_create_group(MPI_COMM_WORLD, evens, 9, &c);
      int sum = 0;
      for (int m : members) sum += m;
      co_await check_members(c, members.size(), sum, __LINE__);
      int crank = -1;
      MPI_Comm_rank(c, &crank);
      C(crank == (int)members.size() - 1 - rank / 2);
      MPI_Comm_free(&c);
    }
    MPI_Group_free(&evens);
    MPI_Group_free(&world);
  }

  long total = 0;
  MPI_Allreduce(&myerr, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    printf("campi comms: %d ranks, errors=%ld: %s\n", size, total, total == 0 ? "PASSED" : "FAILED");
  MPI_Finalize();
  co_return total == 0 ? 0 : 1;
}
