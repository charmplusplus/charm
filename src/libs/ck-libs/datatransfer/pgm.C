#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "fem.h"
#include "charm++.h" /* for CkPrintf */
#include "tetmesh.h"
#include "collidec.h"
#include "paralleltransfer.h"

inline int rankIn(MPI_Comm comm) {
	int ret;
	MPI_Comm_rank(comm,&ret);
	return ret;
}
inline int sizeOf(MPI_Comm comm) {
	int ret;
	MPI_Comm_size(comm,&ret);
	return ret;
}

/**
  Read in this serial .noboite mesh, and partition it over the
  processors.  Returns the local processor's chunk.
*/
int readAndPartition(const char *serial_noboite,MPI_Comm comm) 
{
  int serial_fem=-1;
  if (rankIn(comm)==0) {
    TetMesh serial_mesh;
    readNoboite(fopen(serial_noboite,"r"),serial_mesh);
    CkPrintf("Read input file %s: %d nodes, %d tets\n",
    	serial_noboite,serial_mesh.getPoints(),serial_mesh.getTets());
    serial_fem=FEM_Mesh_allocate();
    writeFEM(serial_fem,serial_mesh);
  }
  int parallel_fem=FEM_Mesh_broadcast(serial_fem,0,comm);
  if (rankIn(comm)==0) FEM_Mesh_deallocate(serial_fem);
  return parallel_fem;
}

/**
  Estimate a good data transfer grid spacing for this mesh.
  Computed as a function of the average tet. edge length.
*/
double estimate_grid_size(const TetMesh &m,MPI_Comm comm) {
  const double edgesPer=10.0;
  double myEst=edgesPer*averageEdgeLength(m), totalEst;
  MPI_Allreduce(&myEst,&totalEst, 1,MPI_DOUBLE, MPI_SUM,comm);
  return totalEst/sizeOf(comm);
}

int main(int argc,char **argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm comm=MPI_COMM_WORLD;
  FEM_Init(comm);

  int myRank=FEM_My_partition();
  double start=FEM_Timer();

// Read in and partition the old mesh:
  int old_fem=readAndPartition("old.noboite",comm);
  TetMesh oldMesh; readFEM(old_fem,oldMesh);

// Read in and partition the new mesh:
  int new_fem=readAndPartition("new.noboite",comm);
  TetMesh newMesh; readFEM(new_fem,newMesh);
  
  if (myRank==0) CkPrintf("[%.3f s] Input files read\n", FEM_Timer()-start);
  
// Fabricate some values on the old mesh:
  double *oldVals=new double[oldMesh.getTets()];
  double *newVals=new double[newMesh.getTets()];
  for (int t=0;t<oldMesh.getTets();t++)
  	oldVals[t]=1*t+10000*myRank;
  
// Set up the search voxel grid:
  const static double gridOrigin[3]={0,0,0};
  double gridSize[3];
  gridSize[0]=gridSize[1]=gridSize[2]=estimate_grid_size(oldMesh,comm);
  if (myRank==0) CkPrintf("[%.3f s] Setting up voxels:\n"
  	"  Origin: (%.3g,%.3g,%.3g), Size: (%.3g,%.3g,%.3g)\n",
	FEM_Timer()-start,
	gridOrigin[0],gridOrigin[1],gridOrigin[2],
	gridSize[0],gridSize[1],gridSize[2]);
  collide_t c=COLLIDE_Init(comm,gridOrigin,gridSize);
  
  if (myRank==0) CkPrintf("[%.3f s] Transferring data.\n",FEM_Timer()-start);
  parallelTransfer(c,comm, 1,
  	oldVals,oldMesh,
	newVals,newMesh);
  
  if (myRank==0) CkPrintf("[%.3f s] Transferred.\n",FEM_Timer()-start);
  
  return 0;
}
