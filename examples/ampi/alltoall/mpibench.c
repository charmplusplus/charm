#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <sys/time.h>
#include <stdlib.h>

#define INIT_SEC 1000000

#define USE_ALLTOALL
//#define USE_BCAST

struct itimerval *tim;
unsigned int Timers = 0;

void   Create_Timers (int n);
void   Start_Timer   (int i, int which);
float  Read_Timer    (int i, int which);

void Create_Timers (int n)
{
  if( Timers > 0 )
    {
      fprintf (stderr, "Create_Timers: timers already created!\n");
      exit (-1);
    }
  
  tim = (struct itimerval*) malloc (n * sizeof (struct itimerval));
  Timers = n;
}

void Start_Timer (int i, int which)
{
  if( i >= Timers )
    {
      fprintf (stderr, "Start_Timers: out-of-range timer index %d\n", i);
      exit (-1);
    }
  
  tim[i].it_value.tv_sec = INIT_SEC;
  tim[i].it_value.tv_usec = 0;
  tim[i].it_interval.tv_sec = INIT_SEC;
  tim[i].it_interval.tv_usec = 0;
  
  setitimer (which, &(tim[i]), NULL);
}

float Read_Timer (int i, int which)
{
  float elapsed_time;
  
  if( i >= Timers )
    {
      fprintf (stderr, "Read_Timer: out-of-range timer index %d\n", i);
      exit (-1);
    }
  
  getitimer (which, &(tim[i]));
  
  elapsed_time = ( (float)INIT_SEC - tim[i].it_value.tv_sec ) -
    ( (float)tim[i].it_value.tv_usec/1000000 );
  
  return elapsed_time;
}

main(int argc, char **argv)
{
  int my_id;		/* process id */
  int p=0;		/* number of processes */
  char* message;	/* storage for the message */
  int i,j, count, k, max_msgs, msg_size;
  float elapsed_time_msec;
  float bandwidth;
  char *sendbuf, *recvbuf;
  int nsteps=100;
  
  MPI_Request *request_recv = (MPI_Request *) malloc(p * sizeof(MPI_Request));
  MPI_Status *status = (MPI_Status *)malloc(p * sizeof(MPI_Status));
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
  MPI_Comm_size( MPI_COMM_WORLD, &p );
  
  if (argc < 2) {
    fprintf (stderr, "need msg size as params\n");
    goto EXIT;
  }
  if ((sscanf (argv[1], "%d", &msg_size) < 1) ){
    fprintf (stderr, "need msg size as params\n");
    goto EXIT;
  }
  max_msgs = 1000; 
  if(argc >2) 
    sscanf (argv[2], "%d", &max_msgs) ;
  
  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD); 
  Create_Timers (1);
  
  sendbuf = (char *)malloc(msg_size * sizeof(char) );
  recvbuf = (char *)malloc(msg_size * sizeof(char) * p);
  
  if(my_id==0) printf("Starting benchmark on %d processors with %d iterations\n", p, max_msgs); 
  Start_Timer (0, ITIMER_REAL);
  
  for(i=0; i<nsteps; i++) {
    for(j=0; j<max_msgs; j++) {
      MPI_Send(sendbuf, msg_size, MPI_CHAR, (my_id+1)%2, j, MPI_COMM_WORLD);
#if 1
      MPI_Recv(recvbuf, msg_size, MPI_CHAR, (my_id+1)%2, j, MPI_COMM_WORLD,&status[j]); 
    }
#else
      MPI_Irecv(recvbuf, msg_size, MPI_CHAR, (my_id+1)%2, j, MPI_COMM_WORLD,&request_recv[j]); 
    }
    MPI_Waitall(max_msgs, request_recv, status); 
#endif
    MPI_Barrier(MPI_COMM_WORLD); 
  }
  
  MPI_Barrier(MPI_COMM_WORLD); 
  if(my_id==0){
    elapsed_time_msec = Read_Timer (0, ITIMER_REAL) * 1000.0 / max_msgs; 
    bandwidth = 2 * 8 * msg_size / (1000.0 * elapsed_time_msec);
    
    fprintf (stdout, "%5d %7d\t ", max_msgs, msg_size);
    fprintf (stdout,"%8.4lf msec,\t %8.3f Mbits/sec\n",
	     elapsed_time_msec/max_msgs, bandwidth);
  }
  free(sendbuf);
  free(recvbuf);
  free(request_recv);
  free(status);
 EXIT:
  MPI_Finalize();
}
