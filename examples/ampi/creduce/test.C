#include <stdio.h>
#include "mpi.h"

int main(int argc,char **argv)
{
	MPI_Init(&argc,&argv);
	double inval,outval;
	int rank,size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	inval = rank+1;
	MPI_Reduce(&inval, &outval, 1, MPI_DOUBLE, MPI_SUM, 
                     0, MPI_COMM_WORLD);
	int expect = (size*(size+1))/2;
	if(rank == 0) {
	  if (outval == expect) 
  	    printf("reduce test passed\n");
  	  else {
  	    printf("reduce test failed!\n");
	    return 1;
          }
	}
	MPI_Finalize();
	return 0;
}
