#include <mpi.h>
#include <stdio.h>

/* This simple application tests 3 PMPI usage scenarios:
 1. PMPI with overridden MPI function (MPI_Send)
 2. PMPI without overridden MPI function (PMPI_Init, PMPI_Comm_size, PMPI_Recv)
 3. Normal MPI functions (MPI_Comm_rank, MPI_Finalize)
*/

/* override MPI_Send(); returns MPI_LASTUSEDCODE+42 instead of MPI_SUCCESS on success.*/
int MPI_Send(const void* buffer, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm)
{
  int result = PMPI_Send(buffer, count, datatype, dest, tag, comm);
  if (result != MPI_SUCCESS) {
    printf("PMPI_Send() FAILED.\n");
    return result;
  }
  else {
    return MPI_LASTUSEDCODE+42; /* value to indicate that our MPI_Send() was called */
  }
}

int main(int argc, char *argv[])
{
  int size, rank;
  int buf[1];
  MPI_Status status;

  PMPI_Init(&argc, &argv);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  buf[0] = 42+rank;

  if (rank == 0 && size<2) {
    printf("Error: %s must be run with 2 or more ranks\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    PMPI_Recv(&buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    if (buf[0] != 42+1) {
      printf("Error: PMPI_Send() FAILED.\n");
      MPI_Finalize();
      return 2;
    }
  }
  else if (rank == 1) {
    int ret = MPI_Send(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (ret != MPI_LASTUSEDCODE+42) {
      printf("PMPI test FAILED.\n");
      MPI_Finalize();
      return 3;
    } else {
      printf("PMPI test passed.\n");
    }
  }

  PMPI_Finalize();
  return 0;
}
