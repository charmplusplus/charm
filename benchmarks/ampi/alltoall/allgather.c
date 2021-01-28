#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  int my_id;     /* process id */
  int p;         /* number of processes */
  char* message; /* storage for the message */
  int i, k, max_msgs, msg_size;
  MPI_Status status; /* return status for receive */
  double startTime = 0;
  double elapsed_time_sec;
  double bandwidth;
  char *sndbuf, *recvbuf;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc < 2)
  {
    fprintf(stderr, "need msg size as params\n");
    goto EXIT;
  }

  if (sscanf(argv[1], "%d", &msg_size) < 1)
  {
    fprintf(stderr, "need msg size as params\n");
    goto EXIT;
  }
  message = (char*)malloc(msg_size);

  max_msgs = 100;
  if (argc > 2) sscanf(argv[2], "%d", &max_msgs);

  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD);

  sndbuf = (char*)malloc(msg_size * sizeof(char));
  recvbuf = (char*)malloc(msg_size * sizeof(char) * p);
  if (my_id == 0)
  {
    int flag = 0;
    printf("Starting benchmark on %d processors with %d iterations\n", p, max_msgs);
    startTime = MPI_Wtime();
  }
  for (i = 0; i < max_msgs; i++)
  {
    MPI_Allgather(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                  MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (my_id == 0)
  {
    elapsed_time_sec = (MPI_Wtime() - startTime) / max_msgs;
    bandwidth = msg_size * (p - 1) / elapsed_time_sec;

    fprintf(stdout, "%5d %7d\t ", max_msgs, msg_size);
    fprintf(stdout, "%8.3f us,\t %8.3f MB/sec\n", elapsed_time_sec * 1e6, bandwidth / 1e6);
  }

  free(sndbuf);
  free(recvbuf);

EXIT:
  MPI_Finalize();
  return 0;
}
