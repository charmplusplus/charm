/* File: fem_mesh_modify.C
 * Authors: Isaac Dooley, Nilesh Choudhury
 * 
 * This file contains a set of functions, which allow primitive operations upon meshes in parallel.
 *
 * See the assumptions listed in fem_mesh_modify.h before using these functions.
 *
 */

#include "fem.h"
#include "fem_impl.h"
#include "fem_mesh_modify.h"

int FEM_add_node(){
  
  // lengthen node array, and any attributes if needed
  // return a new index
  return 0;
}


int FEM_add_shared_node(int* adjacentNodes, int numAdjacentNodes, int upcall){
  // add local node
  int newNode = FEM_add_node();

  // for each adjacent node, if the node is shared
  for(int i=0;i<numAdjacentNodes;i++){
    
    // if node adjacent_nodes[i] is shared,
    {
      // call_shared_node_remote() on all chunks for which the shared node exists
    }



  }


  return 0;
}


// The function called by the entry method on the remote chunk
void FEM_add_shared_node_remote(){
  // create local node
  int newnode = FEM_add_node();
  
  // must negotiate the common IDXL number for the new node, 
  // and store it in appropriate IDXL tables

}




// remove a local or shared node, but NOT a ghost node
void FEM_remove_node(int node){
  
  if(FEM_Is_ghost_index(node))
    CkAbort("Cannot call FEM_remove_node on a ghost node\n");
  
  // if node is shared:
  //   verify it is not adjacent to any elements locally
  //   verify it is not adjacent to any elements on any of the associated chunks
  //   delete it locally and delete it on remote chunks, update IDXL tables

  // if node is local:
  //   verify it is not adjacent to any elements locally
  //   delete it locally
     
}




// Can be called on local or ghost elements
void FEM_remove_element(int element, int elemType){
 
  if(FEM_Is_ghost_index(element)){
    // remove local copy from elem[elemType]->ghost() table
    // call FEM_remove_element_remote on other chunk which owns the element
  }
  else {
    // delete the element from local elem[elem_type] table
  }

  
}

void FEM_remove_element_remote(int element, int elemType){
  // remove local element from elem[elemType] table
}




int FEM_add_element(int* conn, int connSize, int elemType){
  // if no shared or ghost nodes in conn
  //   grow local element and attribute tables if needed
  //   add to the elem[elemType] table
  //   return new element id
  
  // else if any shared nodes but no ghosts in conn
  //   make this element ghost on all others, updating all IDXL's
  //   also in same remote entry method, update adjacencies on all others
  //   grow local element and attribute tables if needed
  //   add to local elem[elemType] table, and update IDXL if needed
  //   update local adjacencies
  //   return the new element id

  // else if any ghosts in conn
  //   promote ghosts to shared on others, requesting new ghosts
  //   grow local element and attribute tables if needed
  //   add to local elem[elemType] table, and update IDXL if needed
  //   update remote adjacencies
  //   update local adjacencies


  return 0;
}


int FEM_add_element_remote(){
  // promote ghosts to shared

  // find new ghosts for remote calling chunk by looking at new shared nodes
  // send these new ghosts to the remote calling chunk.

  // update my adjacencies

  return 0;
}





CProxy_femMeshModify meshMod;

void FEM_REF_INIT(void) {
  CkArrayID femRefId;
  int cid;
  int size;
  TCharm *tc=TCharm::get();

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&cid);
  MPI_Comm_size(comm,&size);
  if(cid==0) {
    CkArrayOptions opts(size);
    opts.bindTo(tc->getProxy()); //bind to the current proxy
    femRefId = CProxy_femMeshModify::ckNew(new femMeshModMsg(size,cid), opts);
  }
  MPI_Bcast(&femRefId, sizeof(CkArrayID), MPI_BYTE, 0, comm);
  meshMod = femRefId;

  femMeshModMsg *fm = new femMeshModMsg(size, cid);
  meshMod[cid].insert(fm);
}


FEM_lock::FEM_lock(int i) {
  idx = i;
  owner = -1;
  isOwner = false;
  isLocked = false;
  hasLocks = false;
}

FEM_lock::~FEM_lock() {
  //before deleting it, ensure that it is not holding any locks
  if(hasLocks) {
    unlock();
  }
  delete &lockedChunks;
}

//will only return if it gets the locks
int FEM_lock::lock(int numNodes, int *nodes, int numElems, int* elems) {
  bool done = false;
  int ret = 0;
  while(!done) {
    if(!isLocked || (isLocked && isOwner)) {
      for(int i=0; i<numNodes; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if not already in it.
	
      }
      //sort the elements in ascending order
      int tmp;
      int numLocks = lockedChunks.size();
      for(int i=0; i<numLocks; i++) {
	for(int j=i+1; j<numLocks; j++) {
	  if(lockedChunks[i] > lockedChunks[j]) {
	    tmp = lockedChunks[i];
	    lockedChunks[i] = lockedChunks[j];
	    lockedChunks[j] = tmp;
	  }
	}
      }
      //lock them
      for(int i=0; i<numLocks; i++) {
	ret = lock(lockedChunks[i],idx);
	if(ret != 1) return -1;
      }
      done = true;
    }
    else {
      CthYield();
      //block
    }
  }
  return 1;
}

int FEM_lock::unlock() {
  bool done = false;
  int ret = 0;
  while(!done) {
    if(!isLocked || (isLocked && isOwner)) {
      //get rid of the locks
      if(hasLocks) {
	for(int i=0; i<lockedChunks.size(); i++) {
	  ret = unlock(lockedChunks[i],idx);
	  if(ret != 1) return -1;
	}
      }
      hasLocks = false;
    }
    else {
      CthYield();
      //block
    }
  }
  return 1;
}

int FEM_lock::lock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  while(true) {
    if(!isLocked || (chunkNo != idx)) {
      if(chunkNo == idx) {
	isLocked = true;
	owner = own;
	if(owner == idx) isOwner = true;
	else isOwner = false;
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	if(ret->i != 1) return -1;
      }
      break;
    }
    else {
      CthYield();
    }
  }
  return 1;
}

//for sanity, only the owner should unlock it
int FEM_lock::unlock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  while(true) {
    if(!isLocked && (chunkNo == idx)) {
      CkError("%d trying to unlock %d which is not locked!!\n",own,chunkNo);
      return -1;
    }
    else if(isLocked && (chunkNo == idx) && (owner != own)) {
      CkError("%d trying to unlock %d which is locked by %d!!\n",own,chunkNo,owner);
      return -1;
    }
    else if(isLocked || (chunkNo != idx)) {
      if(chunkNo == idx) {
	isLocked = false;
	owner = -1;
	isOwner = false;
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	if(ret->i != 1) return -1;
      }
      break;
    }
    else {
      CthYield();
    }
  }
  return 1;
}


femMeshModify::femMeshModify(femMeshModMsg *fm) {
  numChunks = fm->numChunks;
  idx = fm->myChunk;
  fmLock = new FEM_lock(idx);
}

femMeshModify::~femMeshModify() {
  if(fmLock != NULL) {
    delete fmLock;
  }
}

intMsg *femMeshModify::lockRemoteChunk(int2Msg *msg) {
  intMsg *imsg = new intMsg(0);
  int ret = fmLock->lock(msg->i, msg->j);
  imsg->i = ret;
  return imsg;
}

intMsg *femMeshModify::unlockRemoteChunk(int2Msg *msg) {
  intMsg *imsg = new intMsg(0);
  int ret = fmLock->unlock(msg->i, msg->j);
  imsg->i = ret;
  return imsg;
}


#include "femMeshModify.def.h"
