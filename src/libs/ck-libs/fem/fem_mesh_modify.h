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

#include "tcharm.h"
#include "charm++.h"
#include "charm-api.h"
#include "cklists.h"
#include "mpi.h"
#include "fem_mesh.h"
#include "idxl.h"
#include "FEMMeshModify.decl.h"

extern CProxy_femMeshModify meshMod;

// The internal functions which take in a FEM_Mesh*
int FEM_add_node(FEM_Mesh *m, int* adjacent_nodes, int num_adjacent_nodes, int upcall);
void FEM_remove_node(FEM_Mesh *m, int node);
void FEM_remove_element(FEM_Mesh *m, int element, int elem_type);
int FEM_add_element(FEM_Mesh *m, int* conn, int conn_size, int elem_type);
void FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts);
void FEM_Modify_Unlock(FEM_Mesh *m);

CDECL int FEM_add_node(int mesh, int* adjacent_nodes, int num_adjacent_nodes, int upcall){
  return FEM_add_node(FEM_Mesh_lookup(mesh,"FEM_add_node"), adjacent_nodes, num_adjacent_nodes, upcall);}
CDECL void FEM_remove_node(int mesh,int node){
  FEM_remove_node(FEM_Mesh_lookup(mesh,"FEM_remove_node"), node);}
CDECL void FEM_remove_element(int mesh, int element, int elem_type){
  FEM_remove_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type);}
CDECL int FEM_add_element(int mesh, int* conn, int conn_size, int elem_type){
  return FEM_add_element(FEM_Mesh_lookup(mesh,"FEM_add_element"), conn, conn_size, elem_type);}
CDECL void FEM_Modify_Lock(int mesh, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts){
  FEM_Modify_Lock(FEM_Mesh_lookup(mesh,"FEM_Modify_Lock"), affectedNodes, numAffectedNodes, affectedElts, numAffectedElts);}
CDECL void FEM_Modify_Unlock(int mesh){
  FEM_Modify_Unlock(FEM_Mesh_lookup(mesh,"FEM_Modify_Unlock"));}

  
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
  void getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share **chunks);
  bool isShared(int index);
  void splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between, int idxbase);
  void splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between, int idxbase);
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

  sharedNodeMsg(int c, int nB, int *B) {
    chk = c;
    nBetween = nB;
    between = (int *)malloc(nBetween*sizeof(int));
    for(int i=0; i<nBetween; i++) {
      between[i] = B[i];
    }
  }

  ~sharedNodeMsg() {
    if(between) {
      delete between;
    }
  }
};

class femMeshModify/* : public TCharmClient1D */{
  friend class FEM_lock;
  friend class FEM_MUtil;
  friend class FEM_Mesh;

 protected:
  int numChunks;
  int idx;
  FEM_Mesh *fmMesh;
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

  void addSharedNodeRemote(sharedNodeMsg *fm);

};


#endif

