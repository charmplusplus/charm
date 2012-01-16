/***********************************************
          MPI / AMPI pingpong test program
          Prints bandwidth and latency for a specific message size
       
          Sameer Kumar 02/08/05
**************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define     CACHELINE       64
#define     L3CACHE         32*1024*1024
int main(int argc, char **argv)
{
  int my_id;		/* process id */
  int p;		/* number of processes */
  char* message_s, *message_r;	/* storage for the message */
  int i, max_msgs, msg_size;
  MPI_Status status;	/* return status for receive */
  double elapsed_time_sec;
  double bandwidth;
  double startTime = 0;
  int    stride     = CACHELINE;
  int    start;
  MPI_Init( &argc, &argv );
  
  MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
  MPI_Comm_size( MPI_COMM_WORLD, &p );
  
  if ((sscanf (argv[1], "%d", &max_msgs) < 1) ||
      (sscanf (argv[2], "%d", &msg_size) < 1)) {
    fprintf (stderr, "need msg count and msg size as params\n");
    goto EXIT;
  }

  if(msg_size%CACHELINE == 0)
      stride = msg_size;
  else
      stride = ((msg_size/CACHELINE)+1)*stride;

  message_s = (char*)malloc (L3CACHE);  
  message_r = (char*)malloc (L3CACHE);

  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD); 
  
  start = 0;
  if( my_id < p/2 ) {
      startTime = MPI_Wtime();
    
    for(i=0; i<max_msgs; i++){
      MPI_Send(message_s+start, msg_size, MPI_CHAR, my_id+p/2, 0, MPI_COMM_WORLD);
      MPI_Recv(message_s+start, msg_size, MPI_CHAR, my_id+p/2, 0, MPI_COMM_WORLD, 
	       &status);
      start = start+stride;
      if(start ==(L3CACHE-1))
          start = 0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 

    elapsed_time_sec = MPI_Wtime() - startTime; 

    fprintf(stdout, "Totaltime: %8.3f s\n",elapsed_time_sec);
    elapsed_time_sec /= 2;  /* We want the ping performance not round-trip. */
    elapsed_time_sec /= max_msgs; /* time for each message */
    bandwidth = msg_size / elapsed_time_sec; /* bandwidth */
    
    fprintf (stdout, "%5d %7d\t ", max_msgs, msg_size);
    fprintf (stdout,"%8.3f us,\t %8.3f MB/sec\n",
	     elapsed_time_sec * 1e6, bandwidth / 1e6);
    
  }
  else {
    for(i=0; i<max_msgs; i++){
      MPI_Recv(message_r+start, msg_size, MPI_CHAR, my_id-p/2, 0, MPI_COMM_WORLD, 
	       &status); 
      MPI_Send(message_s+start, msg_size, MPI_CHAR, my_id-p/2, 0, MPI_COMM_WORLD);
      start = start+stride;
      if(start ==(L3CACHE-1))
          start = 0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
  }	    
  
  free(message_s);
  //free(message_r);
 EXIT:
  MPI_Finalize();
  
  return 0;
}

