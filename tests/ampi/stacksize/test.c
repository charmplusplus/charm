/*
  Verify that we can use at least 400KB of stack
space under AMPI.  Programs segfault if they run
out of stack space, so it's important to verify that
TCHARM is allocating stacks properly.

The default stack size is 1MB (tcharm.C:tcharm_stacksize);
this program should segfault if run with +tcharm_stacksize=300000.

Orion Sky Lawlor, olawlor@acm.org, 2003/8/25
*/
#include <stdio.h>
#include "mpi.h"

/**
 This recursive procedure consumes at least useKB 
 kilobytes of stack space, by allocating one KB
 and recursing.  The test fails if the routine
 hits a segfault, at which time you can examine 
 nKB in a debugger or core image to see how big 
 the stack actually was.
*/
int testStack(int nKB,int useKB) {
	int i;
#define buf1KB (1024-sizeof(i))
	char buf[buf1KB];
	for (i=0;i<buf1KB;i++)
		buf[i]=0;
	if (nKB>=useKB) 
		return buf[1];
	else
		return buf[buf1KB/2]+testStack(nKB+1,useKB);
}

int main(int argc,char **argv)
{
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	testStack(0,400);
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) {
  	  printf("stacksize test passed\n");
	}
	MPI_Finalize();
	return 0;
}
