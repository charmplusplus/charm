/**
External interface file to 
Charm++ collision detection subsystem
*/
#ifndef __UIUC_CHARM_COLLIDE_H_
#define __UIUC_CHARM_COLLIDE_H_

#ifdef __cplusplus
extern "C" {
#endif


/**
Call this once at system-init time from processor 0.  
It sets up the collision system.
  @param gridSize gives the size of one voxel, which should
   be several times larger than the size of the average object.
  @param gridStart gives the origin of the voxel array--the corner
   of the voxel (0,0,0).  For best performance, if possible you should
   align the voxel array so most objects lie in exactly one object.
*/
void CollideInit(const double *gridStart,const double *gridSize);

/**
Every virtual processor (or FEM chunk) should register before
ANY virtual processor makes a Collide call.
*/
void CollideRegister(int chunkNo);
/**
Every virtual processor should call this before (potentially)
leaving a processor. That is, for migration, you should do:
     CollideUnregister(myChunk);
     AMPI_Migrate();
     CollideRegister(myChunk);
*/
void CollideUnregister(int chunkNo);

/**
Collide these boxes (boxes[0..6*nBox]).
This is a collective call--all registered chunks should make this call.
Unliked CollideClassed, below, no collisions are ignored; or
equivalently, every box lies in its own collision class.
  @param nBoxes number of independent objects to collide.
  @param boxes an array of nBox 3d bounding boxes, stored as 
     x-min, x-max, y-min, y-max, z-min, z-max.
*/
void Collide(int chunkNo,int nBox,const double *boxes);

/**
Collide these boxes (boxes[0..6*nBox]), using the given classes.
void CollideClassed(int chunkNo,int nBox,const double *boxes, const int *classes);
*/

/**
Immediately after a collision, get the number of collisions.
This value is normally used to allocate the array passed to CollideList.
*/
int CollideCount(int chunkNo);

/**
Immediately after a collision, get the colliding boxes (collisions[0..3*nColl]).
   collisions[3*c+0] lists the number of box A, which is always from my chunk
   collisions[3*c+1] lists the source chunk of box B (possibly my own chunk)
   collisions[3*c+2] lists the number of box B on its source chunk
*/
void CollideList(int chunkNo,int *out);

#ifdef __cplusplus
};
#endif

#endif
