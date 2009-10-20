/**
Threaded C interface to 
Charm++ Collision detection subsystem
*/
#ifndef __UIUC_CHARM_COLLIDEC_H_
#define __UIUC_CHARM_COLLIDEC_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
A collide_t is a handle to a single collision detection grid.
*/
typedef int collide_t;

/**
Create a new Collision grid. 

This collective creation call must be made from all the threads of 
a TCHARM array.  

  @param mpi_comm is the MPI communicator, or 0 if not using MPI.
  @param gridSize gives the size of one voxel, which should
   be several times larger than the size of the average object.
  @param gridStart gives the origin of the voxel array--the corner
   of the voxel (0,0,0).  For best performance, if possible you should
   align the voxel array so most objects lie in exactly one voxel.
*/
collide_t COLLIDE_Init(int mpi_comm,
	const double *gridStart,const double *gridSize);

/**
Collide these boxes (boxes[0..6*nBox]).
This is a collective call--all registered chunks should make this call.
Unliked CollideClassed, below, no Collisions are ignored; or
equivalently, every box lies in its own Collision class.
  @param nBoxes number of independent objects to collide.
  @param boxes an array of nBox 3d bounding boxes, stored as 
     x-min, x-max, y-min, y-max, z-min, z-max.
*/
void COLLIDE_Boxes(collide_t c,int nBox,const double *boxes);

/**
Collide these boxes (boxes[0..6*nBox]), using the given box
priorities (prio[0..nBox]).
*/
void COLLIDE_Boxes_prio(int chunkNo,int nBox,const double *boxes, 
	const int *prio);

/**
Immediately after a Collision, get the number of Collision records.
This value is normally used to allocate the array passed to COLLIDE_List.
*/
int COLLIDE_Count(collide_t c);

/**
Immediately after a Collision, get the colliding records into 
   Collisions[0..3*nColl].
   Collisions[3*c+0] lists the number of box A, which is always from my chunk
   Collisions[3*c+1] lists the source chunk of box B (possibly my own chunk)
   Collisions[3*c+2] lists the number of box B on its source chunk
*/
void COLLIDE_List(collide_t c,int *Collisions3);

/**
Destroy this Collision grid.
*/
void COLLIDE_Destroy(collide_t c);

#ifdef __cplusplus
}
#endif

#endif
