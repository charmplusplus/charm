/* File: fem_lock.h
 * Authors: Nilesh Choudhury
 *
 */

#ifndef __CHARM_FEM_BLOCK_H
#define __CHARM_FEM_BLOCK_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"

#define _LOCKCHUNKS

class femMeshModify;

//there is one fem_lock associated with every FEM_Mesh.
class FEM_lock {
  int idx;
  int owner;
  bool isOwner;
  bool isLocked;
  bool hasLocks;
  bool isLocking;
  bool isUnlocking;
  CkVec<int> lockedChunks;
  femMeshModify *mmod;

 private:
  bool existsChunk(int index);

 public:
  FEM_lock() {};
  FEM_lock(int i, femMeshModify *m);
  ~FEM_lock();

  //locks all chunks which contain all the nodes and elements that are passed 
  //in this function
  //locking of the chunks is blocking and is strictly in ascending order.
  int lock(int numNodes, int *nodes, int numElems, int* elems, int elemType=0);
  //unlock all the concerned chunks.
  //since at one point of time one chunk can only lock one set of chunks for
  //one operation, one does not need to pass arguments to unlock.
  int unlock();
  int lock(int chunkNo, int own);
  int unlock(int chunkNo, int own);
  int getIdx() { return idx; }
};

#endif
