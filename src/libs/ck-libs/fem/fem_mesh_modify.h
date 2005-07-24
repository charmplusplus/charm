/*!
 
This file contains a set of functions, which allow primitive operations upon meshes in parallel. The functions are defined in fem_mesh_modify.C.


Assumptions:

The mesh must be in a consistant state before and after these operations:
    - Any shared node must be in the IDXL table for both the local and 
      other chunks.
    - FEM_add_element can only be called with a set of existing local or shared nodes
    - The mesh must be nice. Each element may be adjacent to at most one other element per face/edge/tuple
	- The mesh must have e2e, n2n, and n2e adjacencies computed before any of these functions are called.
	  The calls will maintain the adjacency tables, both remotely and locally.
	- Exactly one ghost layer exists around all chunks before these modification functions are called. 
	  A ghost element is one that is adjacent to at least one shared node. A ghost node is any non-shared 
	  node adjacent to a ghost element. The e2e adjacencies need not have the same definition for 
	  adjacent elements.

 */

#ifndef _FEM_REF_
#define _FEM_REF_

#include "tcharm.h"
#include "charm++.h"
#include "charm-api.h"
#include "cklists.h"
#include "mpi.h"
#include "fem_mesh.h"
#include "fem_adapt_new.h"
#include "idxl.h"
#include "FEMMeshModify.decl.h"

extern CProxy_femMeshModify meshMod;


// The internal functions which take in a FEM_Mesh*, but could feasibly be used by others
int FEM_add_node(FEM_Mesh *m, int* adjacent_nodes=0, int num_adjacent_nodes=0, int upcall=0);
void FEM_remove_node(FEM_Mesh *m, int node);
void FEM_remove_element(FEM_Mesh *m, int element, int elem_type=0);
int FEM_add_element(FEM_Mesh *m, int* conn, int conn_size, int elem_type=0);
void FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes=0, int numAffectedNodes=0, int* affectedElts=0, int numAffectedElts=0, int elemtype=0);
void FEM_Modify_Unlock(FEM_Mesh *m);

// Internal functions which shouldn't be used by anyone else
int FEM_add_node_local(FEM_Mesh *m, int addGhost=0);



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
  int lock(int numNodes, int *nodes, int numElems, int* elems, int elemType=0);
  //unlock all the concerned chunks.
  //since at one point of time one chunk can only lock one set of chunks for
  //one operation, one does not need to pass arguments to unlock.
  int unlock();
  int lock(int chunkNo, int own);
  int unlock(int chunkNo, int own);
  int getIdx() { return idx; }
};

class FEM_MUtil {
  int idx;
  femMeshModify *mmod;

 public:
  FEM_MUtil() {}
  FEM_MUtil(int i, femMeshModify *m);
  ~FEM_MUtil();

  int getIdx() { return idx; }
  //the entType signifies what type of entity to lock. node=0, elem=1;
  //entNo signifies the local index of the entity
  //numChunks is the number of chunks that need to be locked to lock that entity
  //chunks identifies the chunks that need to be locked
  void getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share ***chunks, int elemType=0);
  bool isShared(int index);
  void splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between, int idxbase);
  void splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between, int idxbase);
  void removeNodeAll(FEM_Mesh *m, int localIdx);
  void removeNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  int exists_in_IDXL(FEM_Mesh *m, int localIdx, int chk, int type, int elemType=0);
  // IMPLEMENT ME!!!!
  int lookup_in_IDXL(FEM_Mesh *m, int sharedIdx, int fromChk, int type, int elemType=0) { printf("IMPLEMENT ME: FEM_MUtil::lookup_in_IDXL !!!!\n"); return -1; }

  void addGhostElementRemote(FEM_Mesh *m, int chk, int elemType, int numGhostIndices, int *ghostIndices, int numSharedIndices, int *sharedIndices, int connSize);
  chunkListMsg *getChunksSharingGhostNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  void buildChunkToNodeTable(int *nodetype, int sharedcount, int ghostcount, int localcount, int *conn, int connSize, CkVec<int> ***allShared, int *numSharedChunks, CkVec<int> **allChunks, int ***sharedConn);
  void addElemRemote(FEM_Mesh *m, int chk, int elemtype, int connSize, int *conn, int numGhostIndex, int *ghostIndices);
  void removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices);
  void removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype);
  void FEM_Print_n2n(FEM_Mesh *m, int nodeid);
  void FEM_Print_n2e(FEM_Mesh *m, int nodeid);
  void FEM_Print_e2n(FEM_Mesh *m, int eid);
  void FEM_Print_e2e(FEM_Mesh *m, int eid);
};

class femMeshModMsg : public CMessage_femMeshModMsg {
 public:
  int numChunks;
  int myChunk;

  femMeshModMsg() {}
  
  femMeshModMsg(int num, int idx) {
    numChunks = num;
    myChunk = idx;
  }
  
  ~femMeshModMsg() {}
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

class sharedNodeMsg : public CMessage_sharedNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;

  /*sharedNodeMsg(int c, int nB, int *B) {
    chk = c;
    nBetween = nB;
    between = (int *)malloc(nBetween*sizeof(int));
    for(int i=0; i<nBetween; i++) {
      between[i] = B[i];
    }
    }*/

  ~sharedNodeMsg() {
    if(between) {
      delete between;
    }
  }
};

class removeSharedNodeMsg : public CMessage_removeSharedNodeMsg {
 public:
  int chk;
  int index;
};

class addGhostElemMsg : public CMessage_addGhostElemMsg {
 public:
  int chk;
  int elemType;
  int numGhostIndex;
  int *ghostIndices;
  int numSharedIndex;
  int *sharedIndices;
  int connSize;

  ~addGhostElemMsg() {
    if(ghostIndices) {
      delete ghostIndices;
    }
    if(sharedIndices) {
      delete sharedIndices;
    }
  }
};

class chunkListMsg : public CMessage_chunkListMsg {
 public:
  int numChunkList;
  int *chunkList;

  ~chunkListMsg() {
    if(chunkList) {
      delete chunkList;
    }
  }
};

class addElemMsg : public CMessage_addElemMsg {
 public:
  int chk;
  int elemtype;
  int connSize;
  int *conn;
  int numGhostIndex;
  int *ghostIndices;

  ~addElemMsg() {
    if(conn) {
      delete conn;
    }
    if(ghostIndices) {
      delete ghostIndices;
    }
  }
};

class removeGhostElemMsg : public CMessage_removeGhostElemMsg {
 public:
  int chk;
  int elemtype;
  int elementid;
  int numGhostIndex;
  int *ghostIndices;

  ~removeGhostElemMsg() {
    if(ghostIndices) {
      delete ghostIndices;
    }
  }
};

class removeElemMsg : public CMessage_removeElemMsg {
 public:
  int chk;
  int elementid;
  int elemtype;
};


class femMeshModify : public CBase_femMeshModify {
  friend class FEM_lock;
  friend class FEM_MUtil;
  friend class FEM_Mesh;
  friend class FEM_Adapt;

 protected:
  int numChunks;
  int idx;
  FEM_Mesh *fmMesh;
  FEM_Adapt *fmAdapt;
  FEM_lock *fmLock;
  FEM_MUtil *fmUtil;

 public:
  femMeshModify(femMeshModMsg *fm);
  femMeshModify(CkMigrateMessage *m)/* : TCharmClient1D(m) */{};
  ~femMeshModify();

  intMsg *lockRemoteChunk(int2Msg *i2msg);
  intMsg *unlockRemoteChunk(int2Msg *i2msg);

  void setFemMesh(FEMMeshMsg *fm);
  FEM_lock *getfmLock(){return fmLock;}
  FEM_MUtil *getfmUtil(){return fmUtil;}
  FEM_Adapt *getfmAdapt(){return fmAdapt;}

  void addSharedNodeRemote(sharedNodeMsg *fm);
  void removeSharedNodeRemote(removeSharedNodeMsg *fm);

  void addGhostElem(addGhostElemMsg *fm);
  chunkListMsg *getChunksSharingGhostNode(int2Msg *);
  void addElementRemote(addElemMsg *fm);

  void removeGhostElem(removeGhostElemMsg *fm);
  void removeElementRemote(removeElemMsg *fm);

  void refine_flip_element_leb(int fromChk, int propElemT, int propNodeT,
			       int newNodeT, int nbrOpNodeT, 
			       double longEdgeLen);
};


#endif

