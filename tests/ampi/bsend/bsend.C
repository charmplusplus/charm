// Test that MPI_Bsend does not wait for the matching receive (charm issue #4072, item 3).
// Rank 0 sends two large messages with MPI_Bsend, tag 1 then tag 2; rank 1
// receives tag 2 first. A Bsend that blocks until its receive is posted
// deadlocks here.
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 70000

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size, errors = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    if (rank == 0) printf("bsend needs at least 2 ranks\n");
    MPI_Finalize();
    return 1;
  }

  int bufsize = 0;
  MPI_Pack_size(N, MPI_INT, MPI_COMM_WORLD, &bufsize);
  bufsize = 2 * (bufsize + MPI_BSEND_OVERHEAD);
  char *attached = (char *)malloc(bufsize);
  MPI_Buffer_attach(attached, bufsize);

  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  for (int round = 0; round < 3; round++) {
    if (rank == 0) {
      for (int i = 0; i < N; i++) { a[i] = i + round; b[i] = -i - round; }
      MPI_Bsend(a, N, MPI_INT, 1, 1, MPI_COMM_WORLD);
      MPI_Bsend(b, N, MPI_INT, 1, 2, MPI_COMM_WORLD);
      // Buffered sends must have copied the data: reuse the buffers at once
      for (int i = 0; i < N; i++) { a[i] = 12345; b[i] = 54321; }
    } else if (rank == 1) {
      MPI_Status st;
      MPI_Recv(b, N, MPI_INT, 0, 2, MPI_COMM_WORLD, &st);
      MPI_Recv(a, N, MPI_INT, 0, 1, MPI_COMM_WORLD, &st);
      for (int i = 0; i < N; i++) {
        if (a[i] != i + round || b[i] != -i - round) {
          printf("round %d: data mismatch at %d: %d %d\n", round, i, a[i], b[i]);
          errors++;
          break;
        }
      }
    }
  }

  // MPI_Ibsend: the request completes without the receive being posted
  if (rank == 0) {
    MPI_Request req;
    for (int i = 0; i < N; i++) a[i] = 7 * i;
    MPI_Ibsend(a, N, MPI_INT, 1, 3, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Send(&rank, 1, MPI_INT, 1, 4, MPI_COMM_WORLD);
  } else if (rank == 1) {
    int dummy;
    MPI_Recv(&dummy, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(a, N, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < N; i++)
      if (a[i] != 7 * i) { printf("Ibsend data mismatch at %d\n", i); errors++; break; }
  }

  void *detached; int dsize;
  MPI_Buffer_detach(&detached, &dsize);
  free(attached); free(a); free(b);

  int total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0) printf("bsend: All tests passed\n");
    else printf("bsend: %d errors\n", total);
  }
  MPI_Finalize();
  return total != 0;
}
