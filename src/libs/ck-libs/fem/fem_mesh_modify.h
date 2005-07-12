/*!
 
This file contains a set of functions, which allow primitive operations upon meshes in parallel. The functions are defined in fem_mesh_modify.C.


Assumptions:

The mesh must be in a consistant state before and after these operations:
    - Any shared node must be in the IDXL table for both the local and 
      other chunks.
    - Exactly one ghost layer exists around all chunks. See definition below
    - All adjacency tables must be correct before any of these calls. 
      The calls will maintain the adjacency tables, both remotely
      and locally.
    - FEM_add_element can only be called with a set of existing local or shared nodes


A ghost element is one that is adjacent to at least one shared node. A ghost node is any node adjacent to a ghost element, but is itself not a shared node. Thus we have exactly one layer of ghosts.

 */

#ifndef _FEM_REF_
#define _FEM_REF_

#include "charm++.h"
#include "charm-api.h"
#include "cklists.h"
#include "mpi.h"
#include "femMeshModify.decl.h"
#include "fem_mesh.h"

extern CProxy_femMeshModify meshMod;

int FEM_add_node();
int FEM_add_shared_node(int* adjacent_nodes, int num_adjacent_nodes, int upcall);
void FEM_remove_node(int node);

void FEM_remove_element(int element, int elem_type);
int FEM_add_element(int* conn, int conn_size, int elem_type);


void FEM_REF_INIT(void);

//there is one fem_lock associated with every FEM_Mesh.
class FEM_lock {
  int idx;
  int owner;
  bool isOwner;
  bool isLocked;
  bool hasLocks;
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
  int lock(int numNodes, int *nodes, int numElems, int* elems);
  //unlock all the concerned chunks.
  //since at one point of time one chunk can only lock one set of chunks for
  //one operation, one does not need to pass arguments to unlock.
  int unlock();
  int lock(int chunkNo, int own);
  int unlock(int chunkNo, int own);
};

class FEM_MUtil {
  int idx;
  femMeshModify *mmod;
 public:
  FEM_MUtil() {}
  FEM_MUtil(int i, femMeshModify *m);
  ~FEM_MUtil();

  //the entType signifies what type of entity to lock. node=0, elem=1;
  //entNo signifies the local index of the entity
  //numChunks is the number of chunks that need to be locked to lock that entity
  //chunks identifies the chunks that need to be locked
  void getChunkNos(int entType, int entNo, int *numChunks, int *chunks);
  bool isShared(int index);
};

class femMeshModMsg : public CMessage_femMeshModMsg {
 public:
  int numChunks;
  int myChunk;

  femMeshModMsg(int num, int idx) {
    numChunks = num;
    myChunk = idx;
  }

  ~femMeshModMsg() {
  }
};

class intMsg : public CMessage_intMsg {
 public:
  int i;

  intMsg(int n) {
    i = n;
  }

  ~intMsg(){}
};

class int2Msg : public CMessage_int2Msg {
 public:
  int i, j;

  int2Msg(int m, int n) {
    i = m;
    j = n;
  }

  ~int2Msg(){}
};

class FEMMeshMsg : public CMessage_FEMMeshMsg {
 public:
  FEM_Mesh *m;

  FEMMeshMsg(FEM_Mesh *mh) {
    m = mh;
  }

  ~FEMMeshMsg() {}
};

class femMeshModify {
  friend class FEM_lock;
  friend class FEM_MUtil;
 protected:
  int numChunks;
  int idx;
  FEM_Mesh *fmMesh;
  FEM_lock *fmLock;
  FEM_MUtil *fmUtil;

 public:
  femMeshModify(femMeshModMsg *fm);
  femMeshModify(CkMigrateMessage *m) {};
  ~femMeshModify();

  void setFemMesh(FEMMeshMsg *fm);
  intMsg *lockRemoteChunk(int2Msg *i2msg);
  intMsg *unlockRemoteChunk(int2Msg *i2msg);
};


#endif

