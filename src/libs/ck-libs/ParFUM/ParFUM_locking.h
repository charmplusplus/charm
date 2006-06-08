/* File: ParFUM_locking.h
 * Authors: Nilesh Choudhury
 *
 */

#ifndef __PARFUM_LOCKING_H
#define __PARFUM_LOCKING_H

#define _LOCKCHUNKS

///there is one fem_lock associated with every FEM_Mesh: Chunk Lock (no longer in use)
class FEM_lock {
  ///Index of the lock (chunk)
  int idx;
  ///Current owner of the lock
  int owner;
  ///Is there an owner for the lock
  bool isOwner;
  ///Is the chunk locked
  bool isLocked;
  ///Does this chunk have locks
  bool hasLocks;
  ///Is this chunk locking some chunks
  bool isLocking;
  ///Is this chunk unlocking some chunks
  bool isUnlocking;
  ///The list of chunks locked by this chunk
  CkVec<int> lockedChunks;
  ///cross-pointer to the femMeshModify object
  femMeshModify *mmod;

 private:
  ///Is chunk 'index' locked by this chunk currently
  bool existsChunk(int index);

 public:
  ///default constructor
  FEM_lock() {};
  ///constructor
  FEM_lock(int i, femMeshModify *m);
  ///constructor
  FEM_lock(femMeshModify *m);
  ///destructor
  ~FEM_lock();
  ///Pup this object
  void pup(PUP::er &p);

  ///Return the index of this chunk
  int getIdx() { return idx; }

  ///locks all chunks which contain all the following nodes and elements
  int lock(int numNodes, int *nodes, int numElems, int* elems, int elemType=0);
  ///unlock all the chunks this chunk has locked currently
  int unlock();
  ///chunk 'own' locks the chunk 'chunkNo'
  int lock(int chunkNo, int own);
  ///chunk 'own' unlocks the chunk 'chunkNo'
  int unlock(int chunkNo, int own);
};


///there is one fem_lock associated with every node (locks on elements are not required)
class FEM_lockN {
  ///owner of the lock on this node
  int owner;
  ///Is there some operation waiting for this lock
  int pending;
  ///cross-pointer to the femMeshModify object on this chunk
  femMeshModify *theMod;
  ///index of the node which this lock protects
  int idx;
  ///the number of read locks on this node
  int noreadLocks;
  ///the number of write locks on this node
  int nowriteLocks;
  
 public:
  ///default constructor
  FEM_lockN() {};
  ///constructor
  FEM_lockN(int i,femMeshModify *mod);
  ///constructor
  FEM_lockN(femMeshModify *mod);
  ///destructor
  ~FEM_lockN();
  ///Pup routine for this object
  void pup(PUP::er &p);
  ///Set the femMeshModify object for this chunk
  void setMeshModify(femMeshModify *mod);

  ///return the index of this node
  int getIdx() { return idx; }
  ///reset the data on this node
  void reset(int i,femMeshModify *mod);

  ///get a read lock
  int rlock();
  ///unlock a read lock
  int runlock();
  ///'own' chunk gets a write lock on this node
  int wlock(int own);
  ///unlock the write lock on this node by chunk 'own'
  int wunlock(int own);

  ///Are there any locks on this node
  bool haslocks();
  ///verify if locks exist on this node
  bool verifyLock(void);
  ///who owns this lock now
  int lockOwner();
};

// end ParFUM_locking.h

#endif
