/***********************************************
  MPI / AMPI pingpong test program
  Prints bandwidth and latency for a specific message size

  Sameer Kumar 02/08/05
 **************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void run(int msg_size, int iter, int my_id, int p, char* message_s, char* message_r,
         int printFormat);

int main(int argc, char** argv)
{
  int my_id;                   /* process id */
  int p;                       /* number of processes */
  char *message_s, *message_r; /* storage for the message */
  int max_msg_size, min_msg_size, msg_size, low_iter, high_iter, iter, printFormat;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc < 6)
  {
    if (my_id == 0)
    {
      printf(
          "Doesn't have required input params. Usage: ./pingpong <min-msg-size> "
          "<max-msg-size> <low-iter> <high-iter> <print-format (0 for csv, 1 for "
          "regular)>\n");
    }

    MPI_Finalize();
    return -1;
  }
  else
  {
    min_msg_size = atoi(argv[1]);
    max_msg_size = atoi(argv[2]);
    low_iter = atoi(argv[3]);
    high_iter = atoi(argv[4]);
    printFormat = atoi(argv[5]);
  }

  message_s = (char*)calloc(max_msg_size, 1);
  message_r = (char*)calloc(max_msg_size, 1);

  msg_size = min_msg_size;

  if (my_id < p / 2)
  {
    if (printFormat == 0)
      fprintf(stdout, "Msg Size, Iterations, One-way Time (us), Bandwidth (bytes/us)\n");
    else
      fprintf(stdout, "%-30s %-25s %-20s %-20s\n", "Msg Size", "Iterations",
              "One-way Time (us)", "Bandwidth (bytes/us)");
  }

  while (msg_size <= max_msg_size)
  {
    if (msg_size <= 1048576)
      iter = low_iter;
    else
      iter = high_iter;
    run(msg_size, iter, my_id, p, message_s, message_r, printFormat);
    msg_size *= 2;
  }

  free(message_s);
  free(message_r);

  MPI_Finalize();
  return 0;
}

void run(int msg_size, int iter, int my_id, int p, char* message_s, char* message_r,
         int printFormat)
{
  double elapsed_time_sec;
  double bandwidth;
  double startTime = 0;
  MPI_Status status; /* return status for receive */
  int i;

  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD);

  if (my_id < p / 2)
  {
    startTime = MPI_Wtime();

    for (i = 0; i < iter; i++)
    {
      MPI_Send(message_s, msg_size, MPI_CHAR, my_id + p / 2, 0, MPI_COMM_WORLD);
      MPI_Recv(message_r, msg_size, MPI_CHAR, my_id + p / 2, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    elapsed_time_sec = MPI_Wtime() - startTime;

    elapsed_time_sec /= 2;    /* We want the ping performance not round-trip. */
    elapsed_time_sec /= iter; /* time for each message */
    bandwidth = msg_size / elapsed_time_sec; /* bandwidth */

    if (printFormat == 0)
    {
      fprintf(stdout, "%d,%d,%f,%f\n", msg_size, iter, elapsed_time_sec * 1e6,
              bandwidth / 1e6);
    }
    else
    {
      fprintf(stdout, "%10d\t\t\t%5d\t\t\t", msg_size, iter);
      fprintf(stdout, "%8.3f us\t\t%8.3f MB/sec\n", elapsed_time_sec * 1e6,
              bandwidth / 1e6);
    }
  }
  else
  {
    for (i = 0; i < iter; i++)
    {
      MPI_Recv(message_r, msg_size, MPI_CHAR, my_id - p / 2, 0, MPI_COMM_WORLD, &status);
      MPI_Send(message_s, msg_size, MPI_CHAR, my_id - p / 2, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}
