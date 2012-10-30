/*
Simple TCharm collision detection test program--
	Shows no collisions on a single rank,
	collisions reported for +vp2 and above.

Orion Sky Lawlor, olawlor@acm.org, 2003/3/18.  Updated 2012-07-22 (Public Domain)
 */
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "collidec.h"
#include "bbox.h"

void addObjects(collide_t c,int thisIndex)
{
	CkVector3d o(-6.8,7.9,8.0), x(4.0,0,0), y(0,0.3,0);
	CkVector3d boxSize(0.2,0.2,0.2);
	int nBoxes=2;
	bbox3d *box=new bbox3d[nBoxes];
	for (int i=0;i<nBoxes;i++) {
		CkVector3d c(o+x*thisIndex+y*i);
		CkVector3d c2(c+boxSize);
		box[i].empty();
		box[i].add(c); box[i].add(c2);
	} 
	// first box stretches over into next object:
	box[0].add(o+x*(thisIndex+1.5)+y*2);
	
	COLLIDE_Boxes(c,nBoxes,(const double *)box);
	
	delete[] box;
}

void printCollisions(int myRank,int nColl,int *colls)
{
	printf("**********************************************\n");
	printf("*** %d final collision-- %d records:\n",myRank,nColl);
	int nPrint=nColl;
	const int maxPrint=30;
	if (nPrint>maxPrint) nPrint=maxPrint;
	for (int c=0;c<nPrint;c++) {
		printf("%d:%d hits %d:%d\n",
			myRank,colls[3*c+0],
			colls[3*c+1],colls[3*c+2]);
	}
	if (nPrint!=nColl) printf("<more collisions omitted>");
	printf("**********************************************\n");
}

int main(int argc,char *argv[]) {
	MPI_Init(&argc,&argv);

	int myRank, size;
	MPI_Comm comm=MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&myRank);
	MPI_Comm_size(comm,&size);
	
	const static double gridStart[3]={0,0,0};
	const static double gridSize[3]={2,100,2};
	collide_t c=COLLIDE_Init(comm,gridStart,gridSize);
	
	/** Begin a collision step **/
	addObjects(c,myRank);
	
	int nColl=COLLIDE_Count(c);
	int *colls=new int[3*nColl];
	COLLIDE_List(c,colls);
	printCollisions(myRank,nColl,colls);
	delete[] colls;
	
	return 0;
}

