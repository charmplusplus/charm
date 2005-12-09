/* File: fem_util.h
 * Authors: Nilesh Choudhury
 * 
 */


#ifndef __CHARM_FEM_MUTIL_H
#define __CHARM_FEM_MUTIL_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"

class femMeshModify;
class chunkListMsg;

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
  void splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between);
  void splitEntitySharing(FEM_Mesh *m, int localIdx, int nBetween, int *between, int numChunks, int *chunks);
  void splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between);
  void removeNodeAll(FEM_Mesh *m, int localIdx);
  void removeNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  int exists_in_IDXL(FEM_Mesh *m, int localIdx, int chk, int type, int elemType=0);

  int lookup_in_IDXL(FEM_Mesh *m, int sharedIdx, int fromChk, int type, int elemType=0);
  int getRemoteIdx(FEM_Mesh *m, int elementid, int elemtype);

  void addGhostElementRemote(FEM_Mesh *m, int chk, int elemType, int numGhostIndices, int *ghostIndices, int numSharedIndices, int *sharedIndices, int connSize);
  chunkListMsg *getChunksSharingGhostNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  void buildChunkToNodeTable(int *nodetype, int sharedcount, int ghostcount, int localcount, int *conn, int connSize, CkVec<int> ***allShared, int *numSharedChunks, CkVec<int> **allChunks, int ***sharedConn);
  void addElemRemote(FEM_Mesh *m, int chk, int elemtype, int connSize, int *conn, int numGhostIndex, int *ghostIndices);
  void removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices, int numGhostRNIndex, int *ghostRNIndices, int numGhostREIndex, int *ghostREIndices, int numSharedIndex, int *sharedIndices);
  void removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int permanent);
  void removeGhostNodeRemote(FEM_Mesh *m, int fromChk, int sharedIdx);
  int Replace_node_local(FEM_Mesh *m, int oldIdx, int newIdx);
  void addToSharedList(FEM_Mesh *m, int fromChk, int sharedIdx);
  int eatIntoElement(int localIdx);
  int getLockOwner(int nodeId);
  bool knowsAbtNode(int chk, int nodeId);
  void UpdateGhostSend(int nodeId, int *chunkl, int numchunkl);
  void findGhostSend(int nodeId, int **chunkl, int *numchunkl);

  void StructureTest(FEM_Mesh *m);
  int AreaTest(FEM_Mesh *m);
  int IdxlListTest(FEM_Mesh *m);
  void verifyIdxlListRemote(FEM_Mesh *m, int fromChk, int fsize, int type);
  int residualLockTest(FEM_Mesh *m);

  void FEM_Print_n2n(FEM_Mesh *m, int nodeid);
  void FEM_Print_n2e(FEM_Mesh *m, int nodeid);
  void FEM_Print_e2n(FEM_Mesh *m, int eid);
  void FEM_Print_e2e(FEM_Mesh *m, int eid);
  void FEM_Print_coords(FEM_Mesh *m, int nodeid);

  void idxllock(FEM_Mesh *m, int chk, int type);
  void idxlunlock(FEM_Mesh *m, int chk, int type);
  void idxllockLocal(FEM_Mesh *m, int toChk, int type);
  void idxlunlockLocal(FEM_Mesh *m, int toChk, int type);

  void copyElemData(int etype, int elemid, int newEl);
};

#endif
