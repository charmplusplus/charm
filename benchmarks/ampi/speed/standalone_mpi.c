/**
Call msgspeed routines using bare, native MPI.
Charm isn't involved here at all, so this is a
reasonably honest comparison between Charm's
numbers and those of the real MPI.

Orion Sky Lawlor, olawlor@acm.org, 2003/9/2
*/
#include <stdio.h>
#include <mpi.h>

int main(int argc,char **argv) {
	int verbose=0;
	MPI_Init(&argc,&argv);
	if (argc>1) verbose=atoi(argv[1]);
	startMPItest(MPI_COMM_WORLD,verbose);
}
