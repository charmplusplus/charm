/* File: fem_mesh_modify.h
 * Authors: Nilesh Choudhury
 * 
 */

/**
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

#ifndef __ParFUM_Mesh_Modify_H
#define __ParFUM_Mesh_Modify_H

//stupid number for maximum number of chunks, but reasonable enough
#define MAX_CHUNK 1000000000

///Add a node between some adjacent nodes on this mesh
int FEM_add_node(FEM_Mesh *m, int* adjacent_nodes=0, int num_adjacent_nodes=0, int *chunks=0, int numChunks=0, int forceShared=0);
///Remove a node on this mesh
void FEM_remove_node(FEM_Mesh *m, int node);
///Add an element on this mesh with this connectivity
int FEM_add_element(FEM_Mesh *m, int* conn, int conn_size, int elem_type=0, int chunkNo=-1);
///Remove an element on this mesh
int FEM_remove_element(FEM_Mesh *m, int element, int elem_type=0, int permanent=-1, bool aggressive_node_removal=false);
///Purge the element from this mesh (invalidate entry)
int FEM_purge_element(FEM_Mesh *m, int element, int elem_type=0);

// Internal functions used as helper for the above functions
///Update adjacencies for this new node, and attach a lock to it
int FEM_add_node_local(FEM_Mesh *m, bool addGhost=false, bool doLocking=true, bool doAdjacencies=true);
///Get rid of idxl entries for this node and clear adjacencies, invalidate node
void FEM_remove_node_local(FEM_Mesh *m, int node);
///Update adjacencies for this element and all surrounding nodes/elements
int FEM_add_element_local(FEM_Mesh *m, int *conn, int connSize, int elemType, bool addGhost, bool create_adjacencies=1);
///Clear up the adjacencies
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype);
void FEM_update_new_element_e2e(FEM_Mesh *m, int newEl, int elemType);

///Deprecated: locks all chunks for the nodes and elements specified
int FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes=0, int numAffectedNodes=0, int* affectedElts=0, int numAffectedElts=0, int elemtype=0);
///Deprecated: Unlock all chunks that have been locked by this mesh 
int FEM_Modify_Unlock(FEM_Mesh *m);
///Lock this node on this mesh with a read/write lock
int FEM_Modify_LockN(FEM_Mesh *m, int nodeId, int readLock);
///Lock the read/write lock for this node on this mesh
int FEM_Modify_UnlockN(FEM_Mesh *m, int nodeId, int readLock);
///Reassign the lock on a node when a chunk is losing a node
void FEM_Modify_LockAll(FEM_Mesh*m, int nodeId, bool lockall=true);
///Update the lock on this node (by locking the newly formed node: Deprecated
void FEM_Modify_LockUpdate(FEM_Mesh*m, int nodeId, bool lockall=true);
///For the newly acquired node, correct the lock by removing superfluous locks: Deprecated
void FEM_Modify_correctLockN(FEM_Mesh *m, int nodeId);

///Get the data for 'length' indices from 'fem_mesh' for the 'attr' of 'entity' starting at index 'firstItem'
void FEM_Mesh_dataP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, int datatype,int width);
///Get the data for 'length' indices from 'fem_mesh' for the 'attr' of 'entity' starting at index 'firstItem'
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, IDXL_Layout_t layout);
///Get the data for 'length' indices from 'fem_mesh' for the 'attr' of 'entity' starting at index 'firstItem'
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem,int length, const IDXL_Layout &layout);

///Copy the essential attributes for this ghost node from remote chunks
void FEM_Ghost_Essential_attributes(FEM_Mesh *m, int coord_attr, int bc_attr, int nodeid);



///Message to initialize 'numChunks' and 'chunkIdx' of femMeshModify on all chunks
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

///A Message to encapsulate a boolean
class boolMsg : public CMessage_boolMsg {
 public:
  bool b;

  boolMsg(bool bo) {
    b = bo;
  }

  ~boolMsg() {}
};

///A message to encapsulate an integer
class intMsg : public CMessage_intMsg {
 public:
  int i;

  intMsg(int n) {
    i = n;
  }

  ~intMsg(){}
};

///A message to encapsulate two integers
class int2Msg : public CMessage_int2Msg {
 public:
  int i, j;

  int2Msg(int m, int n) {
    i = m;
    j = n;
  }

  ~int2Msg(){}
};

///A message to encapsulate two doubles
class double2Msg : public CMessage_double2Msg {
 public:
  double i,j;

  double2Msg(double m, double n) {
    i = m;
    j = n;
  }

  ~double2Msg() {}
};

///A message to encapsulate a mesh pointer and a tcharm pointer
class FEMMeshMsg : public CMessage_FEMMeshMsg {
 public:
  FEM_Mesh *m;
  TCharm *t;
	int meshid;

  FEMMeshMsg(FEM_Mesh *mh, TCharm *t1) {
    m = mh;
    t = t1;
  }
  FEMMeshMsg(FEM_Mesh *mh, TCharm *t1,int _meshid) {
    m = mh;
    t = t1;
		meshid = _meshid;
  }

  ~FEMMeshMsg() {}
};

///A message to pack all the data needed to tell a remote chunk to add a new node
class addNodeMsg : public CMessage_addNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;
  int *chunks;
  int numChunks;
  int forceShared;

  ~addNodeMsg() {
    if(between) {
      //delete between;
      //delete chunks;
    }
  }
};

///A message used to tell a remote chunk to add a shared node
class sharedNodeMsg : public CMessage_sharedNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;

  ~sharedNodeMsg() {
    //if(between!=NULL) {
    //delete between;
    //}
  }
};

///A message to tell a remote chunk to remove a shared node
class removeSharedNodeMsg : public CMessage_removeSharedNodeMsg {
 public:
  int chk;
  int index;
};

///A message to tell a remote chunk to add a ghost element (all data is packed)
class addGhostElemMsg : public CMessage_addGhostElemMsg {
 public:
  int chk;
  int elemType;
  int *indices;
  int *typeOfIndex;
  int connSize;

  ~addGhostElemMsg() {
    //if(indices!=NULL) {
    //delete indices;
    //}
    //if(typeOfIndex!=NULL) {
    //delete typeOfIndex;
    //}
  }
};

///A message to return data about the chunks that share/ghost a node/element
class chunkListMsg : public CMessage_chunkListMsg {
 public:
  int numChunkList;
  int *chunkList;
  int *indexList;

  ~chunkListMsg() {
    if(numChunkList>0) {
      //delete chunkList;
      //delete indexList;
    }
  }
};

///A message to pack all data to tell a remote chunk to add an element
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
      //delete conn;
    }
    if(ghostIndices) {
      //delete ghostIndices;
    }
  }
};

///A message to tell a remote chunk to remove a ghost element and some IDXL list entries
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
    if(ghostIndices!=NULL) {
      //   delete ghostIndices;
      //   delete ghostRNIndices;
      //    delete ghostREIndices;
      //    delete sharedIndices;
    }
  }
};

///A message to tell a remote chunk to remove an element
class removeElemMsg : public CMessage_removeElemMsg {
 public:
  int chk;
  int elementid;
  int elemtype;
  int permanent;
  bool aggressive_node_removal;
};

///A message to verify if the IDXL entries for a node/element on one chunk is consistent with another chunk
class verifyghostsendMsg : public CMessage_verifyghostsendMsg {
 public:
  int fromChk;
  int sharedIdx;
  int numchks;
  int *chunks;

  verifyghostsendMsg() {
  }
  
  ~verifyghostsendMsg() {
    //delete chunks;
  }
};

///A message that packs the indices of a bunch of chunks (used for ghost send)
class findgsMsg : public CMessage_findgsMsg {
 public:
  int numchks;
  int *chunks;
  
  ~findgsMsg() {
  }
};

///A message that packs all data for a node/element
class entDataMsg : public CMessage_entDataMsg {
 public:
  char *data;
  int datasize;
  int memsize;

  entDataMsg(int size, int msize) {
    datasize = size;
    memsize = msize;
  }
};

///A message that packs all attributes for a node
class updateAttrsMsg : public CMessage_updateAttrsMsg {
 public:
  char *data;
  int datasize;
  int fromChk;
  int sharedIdx;
  bool isnode;
  bool isGhost;
  int elemType;

  updateAttrsMsg(int size) {
    datasize = size;
    isnode = false;
    isGhost = false;
    elemType = 0;
  }
};




class FEM_Interpolate;

///The shadow array attached to a fem chunk to perform all communication during adaptivity
/** This data structure maintains some adaptivity information, along with
    locks for nodes and idxl lists. It handles all the remote function calls
    and uses FEM_MUtil to perform most of the operations.
 */
class femMeshModify : public CBase_femMeshModify {
  friend class FEM_lock;
  friend class FEM_MUtil;
  friend class FEM_Mesh;
  friend class FEM_Interpolate;
  friend class FEM_Adapt;
  friend class FEM_AdaptL;
  friend class FEM_Adapt_Algs;

 public:
  ///Total number of chunks
  int numChunks;
  ///Index of this chunk (the chunk this is attached to)
  int idx;
  ///The Tcharm pointer to set it even outside the thread..
  TCharm *tc;
  ///The proxy for the current Tcharm object
  CProxy_TCharm tproxy;
  ///cross-pointer to the fem mesh on this chunk
  FEM_Mesh *fmMesh;
  ///Deprecated: used to lock this chunk
  FEM_lock *fmLock;
  ///Set of locks for all nodes on this chunk
  CkVec<FEM_lockN> fmLockN;
  ///Set of locks for all idxl lists
  /** each chunk can have numChunks*5 idxl lists, but numChunks locks. 
   */
  CkVec<bool> fmIdxlLock;
  ///The list of fixed nodes
  /** this list is populated initially, and never changes (defines shape)
   */
  CkVec<int> fmfixedNodes;
  ///Pointer to the utility object (performs most adaptive utilities)
  FEM_MUtil *fmUtil;
  ///Pointer to the object that performs the interpolations
  FEM_Interpolate *fmInp;
  ///Deprecated: Pointer to the FEM_Adapt object
  FEM_Adapt *fmAdapt;
  ///Pointer to the object that performs the primitive adaptive operations
  FEM_AdaptL *fmAdaptL;
  ///Pointer to the object that performs the adaptive algorithms
  FEM_Adapt_Algs *fmAdaptAlgs;

 public:
  ///constructor
  femMeshModify(femMeshModMsg *fm);
  ///constructor for migration
  femMeshModify(CkMigrateMessage *m);
  ///destructor
  ~femMeshModify();

  ///Pup to transfer this object's data
  void pup(PUP::er &p);
  ///This function is overloaded, it is called on this object just after migration
  void ckJustMigrated(void);
  ///Set the mesh pointer after the migration
  void setPointersAfterMigrate(FEM_Mesh *m);

  ///Initialize the mesh pointer for this chunk
  void setFemMesh(FEMMeshMsg *fm);

  ///Deprecated: Try to lock this node on this chunk (the node index is local to this chunk)
  intMsg *lockRemoteChunk(int2Msg *i2msg);
  ///Deprecated: Unlock the node on this chunk (the node index is local to this chunk)
  intMsg *unlockRemoteChunk(int2Msg *i2msg);
  ///Try to lock this node on this chunk (receives a shared/ghost index)
  intMsg *lockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);
  ///Unlock this node on this chunk (receives a shared/ghost index)
  intMsg *unlockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);

  ///Get number of chunks
  int getNumChunks(){return numChunks;}
  ///Get the index of this chunk
  int getIdx(){return idx;}
  ///Get the pointer to the mesh object
  FEM_Mesh *getfmMesh(){return fmMesh;}
  ///Deprecated: Get the pointer to the lock-chunk object
  FEM_lock *getfmLock(){return fmLock;}
  ///Get the status of the lock for this node
  FEM_lockN getfmLockN(int nodeid){
    /*if(!FEM_Is_ghost_index(nodeid)) {
      return fmLockN[nodeid];
      } else {
      return fmgLockN[FEM_To_ghost_index(nodeid)];
      }*/
    CkAssert(nodeid < fmLockN.size());
    return fmLockN[nodeid];
  }
  ///Get the pointer to the utility object on this chunk
  FEM_MUtil *getfmUtil(){return fmUtil;}
  ///Deprecated: Get the pointer to the primitive operation object
  FEM_Adapt *getfmAdapt(){return fmAdapt;}
  ///Get the pointer to the object that encapsulates the primitive adaptive operations
  FEM_AdaptL *getfmAdaptL(){return fmAdaptL;}
  ///Get the pointer to the object that encapsulates the adaptive algorithms
  FEM_Adapt_Algs *getfmAdaptAlgs(){return fmAdaptAlgs;}
  ///Get the pointer to the interpolate object
  FEM_Interpolate *getfmInp(){return fmInp;}

  ///Get the list of chunks that share this node
  chunkListMsg *getChunksSharingGhostNode(int2Msg *);

  ///Add a node on this chunk
  intMsg *addNodeRemote(addNodeMsg *fm);
  ///Add a shared node on this chunk
  void addSharedNodeRemote(sharedNodeMsg *fm);
  ///Remove a shared node on this chunk
  void removeSharedNodeRemote(removeSharedNodeMsg *fm);
  ///Remove a ghost node on this chunk
  void removeGhostNode(int fromChk, int sharedIdx);

  ///Add a ghost element on this chunk
  void addGhostElem(addGhostElemMsg *fm);
  ///Add a local element on this chunk
  intMsg *addElementRemote(addElemMsg *fm);
  ///Remove a ghost element on this chunk
  void removeGhostElem(removeGhostElemMsg *fm);
  ///Remove a local element on this chunk
  void removeElementRemote(removeElemMsg *fm);

  ///Acquire this element from 'fromChk'
  intMsg *eatIntoElement(int fromChk, int sharedIdx);
  
  ///Get the owner of the lock for this shared node
  intMsg *getLockOwner(int fromChk, int sharedIdx);
  ///Does this chunk send this as a ghost to 'toChk'
  boolMsg *knowsAbtNode(int fromChk, int toChk, int sharedIdx);

  ///propagate a flip operation across chunks
  void refine_flip_element_leb(int fromChk, int propElemT, int propNodeT,
			       int newNodeT, int nbrOpNodeT, int nbrghost,
			       double longEdgeLen);

  ///add this node to the shared idxl lists for 'fromChk'
  void addToSharedList(int fromChk, int sharedIdx);
  ///update the attributes of the node with this data
  void updateAttrs(updateAttrsMsg *umsg);
  ///get rid of unnecessary node sends to some chunks for a node
  void updateghostsend(verifyghostsendMsg *vmsg);
  ///find the set of chunks where this node should be sent as a ghost
  findgsMsg *findghostsend(int fromChk, int sharedIdx);
  ///return the shared index on the ghost send list for chunk 'toChk'
  intMsg *getIdxGhostSend(int fromChk, int idxshared, int toChk);

  ///Get the coordinates for this node
  double2Msg *getRemoteCoord(int fromChk, int ghostIdx);
  ///Get the boundary variable for this node
  intMsg *getRemoteBound(int fromChk, int ghostIdx);

  ///find the ghost received from 'transChk' at index 'idxTrans' and add this ghost as received from 'fromChk'
  void updateIdxlList(int fromChk, int idxTrans, int transChk);
  ///remove this node from this idxl list of chunk 'fromChk'
  void removeIDXLRemote(int fromChk, int sharedIdx, int type);
  ///find the shared node with 'transChk' at index 'sharedIdx' and send this as a ghost to 'fromChk'
  void addTransIDXLRemote(int fromChk, int sharedIdx, int type);
  ///verify that the size of the idxl list is same on this chunk and 'fromChk'
  void verifyIdxlList(int fromChk, int size, int type);

  ///lock the idxl list for 'fromChk' on this chunk (blocking call)
  void idxllockRemote(int fromChk, int type);
  ///unlock the idxl list for 'fromChk' on this chunk
  void idxlunlockRemote(int fromChk, int type);

  ///Return the lock owner for this node if there is one (return -1 otherwise)
  intMsg *hasLockRemoteNode(int sharedIdx, int fromChk, int isGhost);
  ///Reassign lock on this node
  void modifyLockAll(int fromChk, int sharedIdx);
  ///verify that this lock is locked on the smallest chunk
  boolMsg *verifyLock(int fromChk, int sharedIdx, int isGhost);
  ///Verify that the number of chunks I get this node as a ghost is same as the number of chunnks that this node is shared on
  void verifyghostsend(verifyghostsendMsg *vmsg);
  ///Should this node be send as a ghost from this chunk to 'fromChk' (if no, return true)
  boolMsg *shouldLoseGhost(int fromChk, int sharedIdx, int toChk);

  ///If this node is not in the ghostSend list add it on the idxl lists on both chunks
  void addghostsendl(int fromChk, int sharedIdx, int toChk, int transIdx);
  ///add this node to the idxl ghost recv list of this chunk
  void addghostsendl1(int fromChk, int transChk, int transIdx);
  ///Add the node as a ghostRecv from 'toChk'
  void addghostsendr(int fromChk, int sharedIdx, int toChk, int transIdx);
  ///Add this node on the ghostSend list with 'fromChk'
  void addghostsendr1(int fromChk, int transChk, int transIdx);
  ///Will it lose this element as a ghost send to 'fromChk'
  boolMsg *willItLose(int fromChk, int sharedIdx);

  ///The element data is copied from the first element to the second
  void interpolateElemCopy(int fromChk, int sharedIdx1, int sharedIdx2);
  ///Remove this ghost element from the ghostRecv idxl list and delete the ghost node
  void cleanupIDXL(int fromChk, int sharedIdx);
  ///Purge this element on this chunk
  void purgeElement(int fromChk, int sharedIdx);
  ///Pack the data from this element/node and return it
  entDataMsg *packEntData(int fromChk, int sharedIdx, bool isnode=false, int elemType=0);
  ///Is this node a fixed node on this chunk
  boolMsg *isFixedNodeRemote(int fromChk, int sharedIdx);

  //debugging helper function
  void finish1(void);
  //debugging helper function
  void finish(void);
};
// end mesh_modify.h

#endif
