#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define NUMTIMES 1
#define REPS 2000
#define MAX_SIZE 131072
#define NUM_SIZES 9

#ifndef MPIWTIME
void getclockvalue(double *retval)
{
  static long zsec = 0;
  static long zusec = 0;
  struct timeval tp;
  struct timezone tzp;

  gettimeofday(&tp, &tzp);

  if ( zsec == 0 ) zsec = tp.tv_sec;
  if ( zusec == 0 ) zusec = tp.tv_usec;

  *retval = (tp.tv_sec - zsec) + (tp.tv_usec - zusec ) * 0.000001 ;
}
#endif

int main(argc,argv)
int argc;
char *argv[];
{
    int myid, root, numprocs, i, j, k, size, num_sizes, times,reps;
    double startwtime, endwtime, opertime[NUMTIMES][NUM_SIZES];
    double mean[NUM_SIZES],min[NUM_SIZES],max[NUM_SIZES];
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int *sendb, *recvb;
    MPI_Status status;
    int msgsizes[NUM_SIZES];

    msgsizes[0] = 2;
    msgsizes[1] = 8;
    msgsizes[2] = 32;
    msgsizes[3] = 128;
    msgsizes[4] = 512;
    msgsizes[5] = 2048;
    msgsizes[6] = 8192;
    msgsizes[7] = 32768;
    msgsizes[8] = 131072;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

#ifdef DEBUG
    fprintf(stdout,"Process %d of %d on %s \n",
	    myid, numprocs, processor_name);
#endif

/* Initialize memory buffers */

    sendb = (int *)malloc(MAX_SIZE*sizeof(int)*numprocs);
    recvb = (int *)malloc(MAX_SIZE*sizeof(int)*numprocs);

#ifdef MPIWTIME
    startwtime = MPI_Wtime();
#else
    getclockvalue (&startwtime);
    getclockvalue (&endwtime);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    for (i=0; i<NUMTIMES; i++)
    {
	for (num_sizes=0; num_sizes<NUM_SIZES; num_sizes++) {
		size = msgsizes[num_sizes];
		for (j=0; j<size*numprocs; j++) {
			sendb[j] = j;
			recvb[j] = 0;
		}
		MPI_Alltoall(sendb,size,MPI_INT,recvb,size,MPI_INT,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef MPIWTIME
		startwtime = MPI_Wtime();
#else
		getclockvalue (&startwtime);
#endif
		for (k=0; k< REPS ; k++) {
			MPI_Alltoall(sendb,size,MPI_INT,recvb,size,MPI_INT,MPI_COMM_WORLD);
		}
#ifdef MPIWTIME
		endwtime = MPI_Wtime();
#else
		getclockvalue (&endwtime);
#endif
		opertime[i][num_sizes] = 
			(endwtime-startwtime)/(float)(REPS) ;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

/* Report results */
    if (myid==0) {
      for (num_sizes=0; num_sizes < NUM_SIZES; num_sizes++) {
	mean[num_sizes] = 0.0;
	min[num_sizes] = 100000. ;
	max[num_sizes] = 0.0 ;
      }

      for (i=0; i<NUMTIMES; i++) {
	for (num_sizes=0; num_sizes < NUM_SIZES; num_sizes++) {
#ifdef DEBUG
		printf("%d %d %g\n",i,msgsizes[num_sizes],
			opertime[i][num_sizes]) ;
#endif
		mean[num_sizes] += opertime[i][num_sizes] ;
		if (min[num_sizes] > opertime[i][num_sizes])
			min[num_sizes] = opertime[i][num_sizes] ;
		if (max[num_sizes] < opertime[i][num_sizes])
			max[num_sizes] = opertime[i][num_sizes] ;
	}
#ifdef DEBUG
	printf("\n");
#endif
      }

      for (num_sizes=0; num_sizes < NUM_SIZES; num_sizes++) 
	mean[num_sizes] /= (float)NUMTIMES;

#ifdef DEBUG
      for (num_sizes=0; num_sizes < NUM_SIZES; num_sizes++) {
	printf("%d %g %g %g\n",msgsizes[num_sizes],mean[num_sizes] * 1000000., 
			min[num_sizes] * 1000000., max[num_sizes] * 1000000. ); 
      }
      printf("================================================\n");
#endif

      times=NUMTIMES; reps=REPS;
      printf("#Alltoall: P=%d, NUMTIMES=%d, REPS=%d\n",
	numprocs, times, reps);
      printf("%d ",numprocs);
      for (num_sizes=0; num_sizes < NUM_SIZES; num_sizes++) 
		printf("%g ",mean[num_sizes] * 1000000.);
      printf("\n");
    }

    MPI_Finalize();
    return 0;
}


