/*  Example code to demonstrate Interoperability between MPI and Charm
    Author - Nikhil Jain
    Contact - nikhil@illinois.edu
*/

//standard header files
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
//header files for libraries in Charm I wish to use with MPI
#include "collidelib/mpicollide.h"
//header file from Charm needed for Interoperation
#include "mpi-interoperate.h"

int main(int argc, char **argv){
  int peid, numpes;
  MPI_Comm newComm;

  //basic MPI initilization
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &peid);
  MPI_Comm_size(MPI_COMM_WORLD, &numpes);


  //initialize Charm for each set
  CharmLibInit(MPI_COMM_WORLD, argc, argv);
  MPI_Barrier(MPI_COMM_WORLD);
  CollisionList *colls;
  CkVector3d o(-6.8,7.9,8.0), x(4.0,0,0), y(0,0.3,0);
  CkVector3d boxSize(0.2,0.2,0.2);
  int nBoxes=1000;
  bbox3d *box=new bbox3d[nBoxes];
  for (int i=0;i<nBoxes;i++) {
	  CkVector3d c(o+x*peid+y*i);
	  CkVector3d c2(c+boxSize);
	  box[i].empty();
	  box[i].add(c); box[i].add(c2);
  }
  // first box stretches over into next object:
  box[0].add(o+x*(peid+1.5)+y*2);
  detectCollision(colls,nBoxes, box, NULL);
  int numColls=colls->length();
  for (int c=0;c<numColls;c++) {
	  printf("%d:%d hits %d:%d\n",
			  (*colls)[c].A.chunk,(*colls)[c].A.number,
			  (*colls)[c].B.chunk,(*colls)[c].B.number);
  }

  delete box;
  MPI_Barrier(MPI_COMM_WORLD);
  CharmLibExit();

  //final synchronization
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;  
}
