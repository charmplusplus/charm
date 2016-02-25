#include <stdio.h>
#include "mpi.h"
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>

#define INIT_SEC 1000000


struct itimerval *tim;
unsigned int Timers = 0;

int  max_msgs = 2;

char hash(int p, int i){
  return (p*(100-p)*770+i*i*3)%127;
}


void   Create_Timers (int n);
void   Start_Timer   (int i, int which);
float  Read_Timer    (int i, int which);

void Create_Timers (int n){
  if( Timers > 0 ){
    fprintf (stderr, "Create_Timers: timers already created!\n");
    exit (-1);
  }
  
  tim = (struct itimerval*) malloc (n * sizeof (struct itimerval));
  Timers = n;
}

void Start_Timer (int i, int which){
  if( i >= Timers ){
    fprintf (stderr, "Start_Timers: out-of-range timer index %d\n", i);
    exit (-1);
  }
  
  tim[i].it_value.tv_sec = INIT_SEC;
  tim[i].it_value.tv_usec = 0;
  tim[i].it_interval.tv_sec = INIT_SEC;
  tim[i].it_interval.tv_usec = 0;
  
  setitimer (which, &(tim[i]), NULL);
}

float Read_Timer (int i, int which){
  float elapsed_time;
  
  if( i >= Timers ){
    fprintf (stderr, "Read_Timer: out-of-range timer index %d\n", i);
    exit (-1);
  }
  
  getitimer (which, &(tim[i]));
  
  elapsed_time = ( (float)INIT_SEC - tim[i].it_value.tv_sec ) -
    ( (float)tim[i].it_value.tv_usec/1000000 );
  
  return elapsed_time;
}

main(int argc, char **argv){
  int my_id;		/* process id */
  int p;		/* number of processes */
  char* message;	/* storage for the message */
  int i, j, k, msg_size;
  MPI_Status status;	/* return status for receive */
  float elapsed_time_msec;
  float bandwidth;
  char *sndbuf, *recvbuf;
  int memory_before, memory_after;
  int memory_diff, local_memory_max;
  int memory_min_small, memory_max_small, memory_min_medium, memory_max_medium, memory_min_normal, memory_max_normal, memory_min_large, memory_max_large;
  
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
  MPI_Comm_size( MPI_COMM_WORLD, &p );
  
  if (argc < 2) {
    fprintf (stderr, "need msg size as params\n");
    goto EXIT;
  }
  
  if(sscanf (argv[1], "%d", &msg_size) < 1){
    fprintf (stderr, "need msg size as params\n");
    goto EXIT;
  }
  message = (char*)malloc (msg_size);

  if(argc>2) 
    sscanf (argv[2], "%d", &max_msgs);


  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD); 
  
  if( my_id == 0 ){
    int flag=0;
  }    
  sndbuf = (char *)malloc(msg_size * sizeof(char) * p);
  recvbuf = (char *)malloc(msg_size * sizeof(char) * p);

  for(j=0;j<p;j++)
	memset(sndbuf+j*msg_size,hash(my_id,j),msg_size);
  memset(recvbuf,0,msg_size*p);
  
  
  
  if(my_id == 0){
	Create_Timers (1);
  }

  // Test Long
  if(1){
	// warm up, not instrumented
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_long(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}

	memset(recvbuf,0,msg_size*p);
	MPI_Barrier(MPI_COMM_WORLD); 
	CmiResetMaxMemory();
	memory_before = CmiMemoryUsage();  // initial memory usage
	MPI_Barrier(MPI_COMM_WORLD); 

	if(my_id == 0){
	  Start_Timer (0, ITIMER_REAL); 
	}
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_long(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	memory_after = CmiMemoryUsage();

	if (CmiMaxMemoryUsage() < memory_before)  
	  local_memory_max = 0;
	else
	  local_memory_max = CmiMaxMemoryUsage() - memory_before;

	// Reduce MAX here
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_max_large, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_min_large, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD));

	if(my_id==0)printf("Large Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\n", (memory_max_large) / 1024, (memory_min_large) / 1024, p, msg_size);

 	for(j=0;j<p;j++)
	  for(k=0;k<msg_size;k++)
		assert(*(recvbuf+j*msg_size+k) == hash(j,my_id) );

  }


  // Test Short
#if 0
  {
	// warm up, not instrumented
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_short(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}

	memset(recvbuf,0,msg_size*p);
	MPI_Barrier(MPI_COMM_WORLD); 
	CmiResetMaxMemory();
	memory_before = CmiMemoryUsage();  // initial memory usage
	MPI_Barrier(MPI_COMM_WORLD); 

	if(my_id == 0){
	  Start_Timer (0, ITIMER_REAL); 
	}
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_short(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	memory_after = CmiMemoryUsage();

	if (CmiMaxMemoryUsage() < memory_before)  
	  local_memory_max = 0;
	else
	  local_memory_max = CmiMaxMemoryUsage() - memory_before;

	// Reduce MAX here
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_max_small, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_min_small, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD));

	if(my_id==0)printf("Small Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\n", (memory_max_small) / 1024, (memory_min_small) / 1024, p, msg_size);

	for(j=0;j<p;j++)
	  for(k=0;k<msg_size;k++)
		assert(*(recvbuf+j*msg_size+k) == hash(j,my_id) );
  }
#endif

  // Test Medium
  if(1){
	// warm up, not instrumented
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_medium(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}

	memset(recvbuf,0,msg_size*p);
	MPI_Barrier(MPI_COMM_WORLD); 
	CmiResetMaxMemory();
	memory_before = CmiMemoryUsage();  // initial memory usage
	MPI_Barrier(MPI_COMM_WORLD); 

	if(my_id == 0){
	  Start_Timer (0, ITIMER_REAL); 
	}
	for(i=0; i<max_msgs; i++) {
	  AMPI_Alltoall_medium(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	memory_after = CmiMemoryUsage();

	if (CmiMaxMemoryUsage() < memory_before)  
	  local_memory_max = 0;
	else
	  local_memory_max = CmiMaxMemoryUsage() - memory_before;

	// Reduce MAX here
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_max_medium, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_min_medium, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD));

	if(my_id==0)	printf("Med   Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\n", (memory_max_medium) / 1024, (memory_min_medium) / 1024, p, msg_size);

	for(j=0;j<p;j++)
	  for(k=0;k<msg_size;k++)
		assert(*(recvbuf+j*msg_size+k) == hash(j,my_id) );
  }

  // Test standard version
  {
	// warm up, not instrumented
	for(i=0; i<max_msgs; i++) {
	  MPI_Alltoall(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}
	
	memset(recvbuf,0,msg_size*p);
	MPI_Barrier(MPI_COMM_WORLD); 
	CmiResetMaxMemory();
	memory_before = CmiMemoryUsage();  // initial memory usage
	MPI_Barrier(MPI_COMM_WORLD); 

	if(my_id == 0){
	  Start_Timer (0, ITIMER_REAL); 
	}
	for(i=0; i<max_msgs; i++) {
	  MPI_Alltoall(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	memory_after = CmiMemoryUsage();

	if (CmiMaxMemoryUsage() < memory_before)  
	  local_memory_max = 0;
	else
	  local_memory_max = CmiMaxMemoryUsage() - memory_before;

	// Reduce MAX here
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_max_normal, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
	assert(MPI_SUCCESS==MPI_Reduce(&local_memory_max, &memory_min_normal, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD));

	
	if(my_id==0)  printf("Norm  Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\n", (memory_max_normal) / 1024, (memory_min_normal) / 1024, p, msg_size);

	for(j=0;j<p;j++)
	  for(k=0;k<msg_size;k++)
		assert(*(recvbuf+j*msg_size+k) == hash(j,my_id) );
  }

  if(my_id==0) printf("\n");


  free(sndbuf);
  free(recvbuf);
  
 EXIT:
  MPI_Finalize();
}

