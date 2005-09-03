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
#include "fem_adapt_lock.h"
#include "fem_adapt_algs.h"
#include "fem_interpolate.h"
#include "fem_lock.h"
#include "fem_lock_node.h"
#include "fem_util.h"
#include "idxl.h"
#include "FEMMeshModify.decl.h"

extern CProxy_femMeshModify meshMod;


// The internal functions which take in a FEM_Mesh*, but could feasibly be used by others
int FEM_add_node(FEM_Mesh *m, int* adjacent_nodes=0, int num_adjacent_nodes=0, int chunkNo=-1, int forceShared=0, int upcall=0);
void FEM_remove_node(FEM_Mesh *m, int node);
int FEM_remove_element(FEM_Mesh *m, int element, int elem_type=0, int permanent=0);
int FEM_add_element(FEM_Mesh *m, int* conn, int conn_size, int elem_type=0, int chunkNo=-1);
int FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes=0, int numAffectedNodes=0, int* affectedElts=0, int numAffectedElts=0, int elemtype=0);
int FEM_Modify_Unlock(FEM_Mesh *m);
int FEM_Modify_LockN(FEM_Mesh *m, int nodeId, int readLock);
int FEM_Modify_UnlockN(FEM_Mesh *m, int nodeId, int readLock);

// Internal functions which shouldn't be used by anyone else
int FEM_add_node_local(FEM_Mesh *m, int addGhost=0);
void FEM_remove_node_local(FEM_Mesh *m, int node);
int FEM_add_element_local(FEM_Mesh *m, const int *conn, int connSize, int elemType, int addGhost);
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype);


void FEM_Mesh_dataP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, int datatype,int width);
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, IDXL_Layout_t layout);
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem,int length, const IDXL_Layout &layout);


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

class double2Msg : public CMessage_double2Msg {
 public:
  double i,j;

  double2Msg(double m, double n) {
    i = m;
    j = n;
  }

  ~double2Msg() {}
};

class FEMMeshMsg : public CMessage_FEMMeshMsg {
 public:
  FEM_Mesh *m;

  FEMMeshMsg(FEM_Mesh *mh) {
    m = mh;
  }

  ~FEMMeshMsg() {}
};

class addNodeMsg : public CMessage_addNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;
  int chunkNo;
  int forceShared;
  int upcall;

  ~addNodeMsg() {
    if(between) {
      delete between;
    }
  }
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
  int numGhostRNIndex;
  int numGhostREIndex;
  int numSharedIndex;
  int *ghostIndices;
  int *ghostRNIndices;
  int *ghostREIndices;
  int *sharedIndices;

  ~removeGhostElemMsg() {
    if(ghostIndices) {
      delete ghostIndices;
      delete ghostRNIndices;
      delete ghostREIndices;
      delete sharedIndices;
    }
  }
};

class removeElemMsg : public CMessage_removeElemMsg {
 public:
  int chk;
  int elementid;
  int elemtype;
  int permanent;
};


class femMeshModify : public CBase_femMeshModify {
  friend class FEM_lock;
  friend class FEM_MUtil;
  friend class FEM_Mesh;
  friend class FEM_Interpolate;
  friend class FEM_Adapt;
  friend class FEM_AdaptL;
  friend class FEM_Adapt_Algs;

 public:
  int numChunks;
  int idx;
  FEM_Mesh *fmMesh;
  FEM_lock *fmLock;
  CkVec<FEM_lockN *> fmLockN;
  //CkVec<FEM_lockN *> *fmgLockN;
  FEM_MUtil *fmUtil;
  FEM_Interpolate *fmInp;
  FEM_Adapt *fmAdapt;
  FEM_AdaptL *fmAdaptL;
  FEM_Adapt_Algs *fmAdaptAlgs;

 public:
  femMeshModify(femMeshModMsg *fm);
  femMeshModify(CkMigrateMessage *m)/* : TCharmClient1D(m) */{};
  ~femMeshModify();

  intMsg *lockRemoteChunk(int2Msg *i2msg);
  intMsg *unlockRemoteChunk(int2Msg *i2msg);
  intMsg *lockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);
  intMsg *unlockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);
  void setFemMesh(FEMMeshMsg *fm);
  int getNumChunks(){return numChunks;}
  int getIdx(){return idx;}
  FEM_Mesh *getfmMesh(){return fmMesh;}
  FEM_lock *getfmLock(){return fmLock;}
  FEM_lockN *getfmLockN(int nodeid){
    /*if(!FEM_Is_ghost_index(nodeid)) {
      return fmLockN[nodeid];
      } else {
      return fmgLockN[FEM_To_ghost_index(nodeid)];
      }*/
    return fmLockN[nodeid];
  }
  FEM_MUtil *getfmUtil(){return fmUtil;}
  FEM_Adapt *getfmAdapt(){return fmAdapt;}
  FEM_AdaptL *getfmAdaptL(){return fmAdaptL;}
  FEM_Adapt_Algs *getfmAdaptAlgs(){return fmAdaptAlgs;}
  FEM_Interpolate *getfmInp(){return fmInp;}

  intMsg *addNodeRemote(addNodeMsg *fm);
  void addSharedNodeRemote(sharedNodeMsg *fm);
  void removeSharedNodeRemote(removeSharedNodeMsg *fm);

  void addGhostElem(addGhostElemMsg *fm);
  chunkListMsg *getChunksSharingGhostNode(int2Msg *);
  void addElementRemote(addElemMsg *fm);

  void removeGhostElem(removeGhostElemMsg *fm);
  void removeElementRemote(removeElemMsg *fm);

  void refine_flip_element_leb(int fromChk, int propElemT, int propNodeT,
			       int newNodeT, int nbrOpNodeT, int nbrghost,
			       double longEdgeLen);

  void addToSharedList(int fromChk, int sharedIdx);
  void updateNodeAttrs(int fromChk, int sharedIdx, double coordX, double coordY, int bound);
  double2Msg *getRemoteCoord(int fromChk, int ghostIdx);
  intMsg *getRemoteBound(int fromChk, int ghostIdx);
};


#endif

