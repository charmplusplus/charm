/**
 * AMPI Migration Test:
 * Migrate AMPI rank 1 from PE to PE in order.
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cassert>
#include "mpi.h"

int get_my_pe() {
  int flag;
  int *my_wth;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &my_wth, &flag);
  assert(flag == 1);
  return *my_wth;
}

int get_num_pes() {
  int flag;
  int *num_wths;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_NUM_WTHS, &num_wths, &flag);
  assert(flag == 1);
  return *num_wths;
}

int main(int argc, char **argv)
{
  int rank;        /* my rank # */
  int p;           /* number of ranks */
  int my_init_pe;  /* my initial PE according to mapfile */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int my_pe = get_my_pe();

  // Test that mapfile is being used correctly:
  if (argc == 2) {
    if (rank == 0) {
      printf("Testing mapfile correctness\n");
      FILE *mapf = fopen("mapfile", "r");
      if (mapf == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::vector<int> init_pes(p);

      for (int i=0; i<p; i++) {
        if (fscanf(mapf, "%d\n", &init_pes[i]) != 1) {
          MPI_Abort(MPI_COMM_WORLD, 2);
        }
      }

      MPI_Scatter(init_pes.data(), 1, MPI_INT, &my_init_pe, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, &my_init_pe, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    assert(my_init_pe == my_pe);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (p >= 1) {
    int num_pes = get_num_pes();
    printf("Begin migrating\n");

    for (int i=0; i<=num_pes; i++) {
      if (rank == 1) {
        int dest_pe = (my_pe + 1) % num_pes;
	printf("Trying to migrate rank %d from PE %d to %d\n", rank, my_pe, dest_pe);
	AMPI_Migrate_to_pe(dest_pe);
        my_pe = get_my_pe();
	printf("After migration of rank %d to PE %d\n", rank, my_pe);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      printf("Rank %d done with step %d\n", rank, i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d done migrating\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank==0) {
    printf("All tests passed\n");
  }

  MPI_Finalize();
  return 0;
}
