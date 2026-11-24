// Test MPI_Group_free on groups used with MPI_Comm_create (charm issue #4072, item 2).
// A group must stay valid, and freeable, independently of the communicators
// created from it or the communicator it was obtained from.
#include <stdio.h>
#include "mpi.h"

static int errors = 0;
#define CHECK(c, what) do { if (!(c)) { printf("[%d] FAILED: %s (line %d)\n", wrank, what, __LINE__); errors++; } } while (0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int wrank, wsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);

  for (int iter = 0; iter < 4; iter++) {
    // 1. Group of a duplicate of WORLD; the dup is freed before the group
    //    (mpich attr/attrt does this).
    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    MPI_Group dupGroup;
    MPI_Comm_group(dup, &dupGroup);
    MPI_Comm full;
    MPI_Comm_create(dup, dupGroup, &full);
    int r = -1, s = -1;
    MPI_Comm_rank(full, &r);
    MPI_Comm_size(full, &s);
    CHECK(r == wrank && s == wsize, "comm created from whole group");

    // 2. Several subset groups alive at once, freed in creation order, so
    //    that freeing one must not invalidate the handles of the others.
    MPI_Group worldGroup, lo, hi, evens;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    int half = wsize / 2;
    int loRange[1][3] = {{0, half - 1, 1}};
    int hiRange[1][3] = {{half, wsize - 1, 1}};
    int evRange[1][3] = {{0, wsize - 1, 2}};
    MPI_Group_range_incl(worldGroup, 1, loRange, &lo);
    MPI_Group_range_incl(worldGroup, 1, hiRange, &hi);
    MPI_Group_range_incl(worldGroup, 1, evRange, &evens);

    MPI_Comm loComm, hiComm, evComm;
    MPI_Comm_create(full, lo, &loComm);
    MPI_Group_free(&lo);
    CHECK(lo == MPI_GROUP_NULL, "Group_free sets MPI_GROUP_NULL");
    MPI_Comm_create(full, hi, &hiComm);
    MPI_Comm_create(full, evens, &evComm);

    int gsize = -1;
    MPI_Group_size(hi, &gsize);
    CHECK(gsize == wsize - half, "size of hi group after freeing lo");
    MPI_Group_size(evens, &gsize);
    CHECK(gsize == (wsize + 1) / 2, "size of evens group after freeing lo");
    MPI_Group_free(&hi);
    MPI_Group_size(evens, &gsize);
    CHECK(gsize == (wsize + 1) / 2, "size of evens group after freeing hi");
    MPI_Group_free(&evens);

    // Use the communicators after their groups are gone
    MPI_Comm mine = (wrank < half) ? loComm : hiComm;
    CHECK(mine != MPI_COMM_NULL, "member of lo or hi comm");
    int sum = -1, expect = 0, base = (wrank < half) ? 0 : half, n = (wrank < half) ? half : wsize - half;
    for (int i = 0; i < n; i++) expect += base + i;
    MPI_Allreduce(&wrank, &sum, 1, MPI_INT, MPI_SUM, mine);
    CHECK(sum == expect, "allreduce on lo/hi comm");
    if (wrank % 2 == 0) {
      CHECK(evComm != MPI_COMM_NULL, "even rank in evens comm");
      MPI_Comm_rank(evComm, &r);
      CHECK(r == wrank / 2, "rank in evens comm");
      MPI_Comm_free(&evComm);
    } else {
      CHECK(evComm == MPI_COMM_NULL, "odd rank not in evens comm");
    }
    if (loComm != MPI_COMM_NULL) MPI_Comm_free(&loComm);
    if (hiComm != MPI_COMM_NULL) MPI_Comm_free(&hiComm);

    // 3. The group of a freed communicator is still usable, then freeable.
    MPI_Comm_free(&full);
    MPI_Comm_free(&dup);
    MPI_Group_size(dupGroup, &gsize);
    CHECK(gsize == wsize, "size of group whose comm was freed");
    MPI_Group_free(&dupGroup);
    MPI_Group_free(&worldGroup);
  }

  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (wrank == 0) {
    if (total == 0) printf("groupfree: All tests passed\n");
    else printf("groupfree: %d errors\n", total);
  }
  MPI_Finalize();
  return total != 0;
}
