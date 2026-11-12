// MPI_Comm_create_group, called by the group's members only: the even ranks
// in reverse order, then all ranks. Each new communicator is checked with an
// MPI_Allreduce over it.
#include <stdio.h>
#include <vector>
#include "mpi.h"

static int errors = 0;

static void check(MPI_Comm world, MPI_Group group, int rank, int tag, const char *what) {
  int grank, gsize;
  MPI_Group_rank(group, &grank);
  MPI_Group_size(group, &gsize);
  if (grank == MPI_UNDEFINED) return;
  std::vector<int> granks(gsize), wranks(gsize);
  MPI_Group wgroup;
  MPI_Comm_group(world, &wgroup);
  for (int i = 0; i < gsize; i++) granks[i] = i;
  MPI_Group_translate_ranks(group, gsize, granks.data(), wgroup, wranks.data());
  MPI_Group_free(&wgroup);
  MPI_Comm c;
  MPI_Comm_create_group(world, group, tag, &c);
  int csize = -1, crank = -1, sum = -1, expect = 0;
  MPI_Comm_size(c, &csize);
  MPI_Comm_rank(c, &crank);
  MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, c);
  for (int w : wranks) expect += w;
  if (csize != gsize || crank != grank || sum != expect) {
    errors++;
    printf("[%d] %s: size %d rank %d sum %d, expected %d %d %d\n", rank, what, csize, crank,
           sum, gsize, grank, expect);
  }
  MPI_Comm_free(&c);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Group world, evens;
  MPI_Comm_group(MPI_COMM_WORLD, &world);
  std::vector<int> members;
  for (int i = (size - 1) / 2 * 2; i >= 0; i -= 2) members.push_back(i);
  MPI_Group_incl(world, members.size(), members.data(), &evens);
  check(MPI_COMM_WORLD, evens, rank, 3, "even ranks");
  check(MPI_COMM_WORLD, world, rank, 4, "all ranks");
  MPI_Group_free(&evens);
  MPI_Group_free(&world);
  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) printf("comm_create_group: %d ranks, errors=%d\n", size, total);
  MPI_Finalize();
  return total != 0;
}
