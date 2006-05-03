/* File: lock.h
 * Authors: Nilesh Choudhury
 *
 */

#define _LOCKCHUNKS

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
  FEM_lock(femMeshModify *m);
  ~FEM_lock();
  void pup(PUP::er &p);

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


// end lock.h


/* File: lock_node.h
 * Authors: Nilesh Choudhury
 * 
 */


//there is one fem_lock associated with every node (locks on elements are not required)
//should lock all nodes, involved in any operation
class FEM_lockN {
  int owner, pending;
  femMeshModify *theMod;
  int idx; //index of the node
  int noreadLocks;
  int nowriteLocks;
  
 public:
  FEM_lockN() {};
  FEM_lockN(int i,femMeshModify *mod);
  FEM_lockN(femMeshModify *mod);
  ~FEM_lockN();
  void pup(PUP::er &p);
  void setMeshModify(femMeshModify *mod);
  void reset(int i,femMeshModify *mod);
  int rlock();
  int runlock();
  int wlock(int own);
  int wunlock(int own);
  bool haslocks();
  bool verifyLock(void);
  int lockOwner();
  int getIdx() { return idx; }
};

// end lock_node.h

