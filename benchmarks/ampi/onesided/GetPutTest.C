/***********************************************
          MPI / AMPI test program for OneSided Put/Get
          Prints bandwidth and latency for a specific message size
          Yan Shi 04/05/2006
**************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
  int my_id,next_id;            /* process id */
  int p;                /* number of processes */
  char* message_s, *message_r;  /* storage for the message */
  int i, j, max_msgs, msg_size;
  MPI_Status status;    /* return status for receive */
  double elapsed_time_sec;
  double bandwidth;
  double startTime = 0;
  MPI_Win win; 

  MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
  MPI_Comm_size( MPI_COMM_WORLD, &p );

  if ((sscanf (argv[1], "%d", &max_msgs) < 1) ||
      (sscanf (argv[2], "%d", &msg_size) < 1)) {
    fprintf (stderr, "need msg count and msg size as params\n");
    goto EXIT;
  }
  fprintf(stdout, "Msg size %d,  loop %d\n", msg_size, max_msgs);
  //allocate data   
  message_s = (char*)malloc (msg_size);
  message_r = (char*)malloc (msg_size);

  //init data for verifying purpose
  message_s[0] = 'a';

  //create window
  MPI_Win_create(message_s, msg_size, 1, 0, MPI_COMM_WORLD, &win);

  //timing
  MPI_Barrier(MPI_COMM_WORLD);
  startTime = MPI_Wtime();

  next_id = my_id+1>=p ? 0 : my_id+1;
  // combined local transpose with global all-to-all
  for(i=0; i<max_msgs; i++){
      MPI_Get(message_r, msg_size, MPI_CHAR, next_id, 0, msg_size, MPI_CHAR, win);
//      fprintf(stdout, "[%d] get %c\n", my_id, message_r[0]);
  }
  MPI_Win_fence(0, win);
  fprintf(stdout, "[%d] get %c\n", my_id, message_r[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time_sec = MPI_Wtime() - startTime;
  fprintf(stdout, "Get Performance:\n");
  fprintf(stdout, "Totaltime: %8.3f s\n",elapsed_time_sec);
  elapsed_time_sec /= max_msgs; //time for each message
  bandwidth = msg_size / elapsed_time_sec; //bandwidth

  fprintf (stdout, "%5d %7d\t ", max_msgs, msg_size);
  fprintf (stdout,"%8.3f us,\t %8.3f MB/sec\n",
           elapsed_time_sec * 1e6, bandwidth / 1e6);

  message_r[0] = 'b';
  startTime = MPI_Wtime();
  for(i=0; i<max_msgs; i++){
      MPI_Put(message_r, msg_size, MPI_CHAR, next_id, 0, msg_size, MPI_CHAR, win);
  }
  MPI_Win_fence(0, win);
  fprintf(stdout, "[%d] get %c\n", my_id, message_s[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time_sec = MPI_Wtime() - startTime;
  fprintf(stdout, "Put Performance:\n");
  fprintf(stdout, "Totaltime: %8.3f s\n",elapsed_time_sec);
  elapsed_time_sec /= max_msgs; //time for each message
  bandwidth = msg_size / elapsed_time_sec; //bandwidth

  fprintf (stdout, "%5d %7d\t ", max_msgs, msg_size);
  fprintf (stdout,"%8.3f us,\t %8.3f MB/sec\n",
           elapsed_time_sec * 1e6, bandwidth / 1e6);

  //deallocate data and stuff
  MPI_Win_free(&win);
  free(message_s);
  free(message_r);

 EXIT:
  MPI_Finalize();
  return 0;
}

