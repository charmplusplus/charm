/**
 * AMPI Migration Test:
 * Migrate AMPI rank 1 from PE to PE in order.
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstring>
#include "mpi.h"

int get_my_pe() {
  int flag;
  int *my_wth;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &my_wth, &flag);
  if (flag != 1) {
    printf("AMPI comm attr AMPI_MY_WTH undefined!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return *my_wth;
}

int get_num_pes() {
  int flag;
  int *num_wths;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_NUM_WTHS, &num_wths, &flag);
  if (flag != 1) {
    printf("AMPI comm attr AMPI_NUM_WTHS undefined!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return *num_wths;
}

int main(int argc, char **argv)
{
  int rank;       /* my rank # */
  int p;          /* number of ranks */
  int my_init_pe; /* my initial PE according to mapfile */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int my_pe = get_my_pe();

  // Test that mapfile is being used correctly:
  if (argc >= 2 && !strcmp(argv[1], "--test_mapfile")) {
    if (rank == 0) {
      printf("Testing mapfile correctness\n");
      FILE *mapf = fopen("mapfile", "r");
      if (mapf == NULL) {
        printf("Missing file named 'mapfile'!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      std::vector<int> init_pes(p);
      for (int i=0; i<p; i++) {
        if (fscanf(mapf, "%d\n", &init_pes[i]) != 1) {
          printf("Unrecognized mapfile formatting!\n");
          fclose(mapf);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
      fclose(mapf);

      MPI_Scatter(init_pes.data(), 1, MPI_INT, &my_init_pe, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, &my_init_pe, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (my_pe != my_init_pe) {
      printf("Rank %d on PE %d should have been initialized on PE %d according to mapfile!\n", rank, my_pe, my_init_pe);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
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
        if (my_pe != dest_pe) {
          printf("Rank %d is on PE %d but should have migrated to PE %d!\n", rank, my_pe, dest_pe);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
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
