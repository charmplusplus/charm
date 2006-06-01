/* File: fem_lock.C
 * Authors: Nilesh Choudhury
 *
 */

/** This entire class is DEPRECATED!
    It provides locking mechanism for chunks
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"

FEM_lock::FEM_lock(int i, femMeshModify *m) {
  idx = i;
  owner = -1;
  isOwner = false;
  isLocked = false;
  hasLocks = false;
  isLocking = false;
  isUnlocking = false;
  lockedChunks.removeAll();
  mmod = m;
}

FEM_lock::FEM_lock(femMeshModify *m) {
  idx = -1;
  owner = -1;
  isOwner = false;
  isLocked = false;
  hasLocks = false;
  isLocking = false;
  isUnlocking = false;
  lockedChunks.removeAll();
  mmod = m;
}

FEM_lock::~FEM_lock() {
  //before deleting it, ensure that it is not holding any locks
  if(hasLocks) {
    unlock();
  }
}

void FEM_lock::pup(PUP::er &p) {
  p|idx;
  p|owner;
  p|isOwner;
  p|isLocked;
  p|hasLocks;
  p|isLocking;
  p|isUnlocking;
  p|lockedChunks;
}



bool FEM_lock::existsChunk(int index) {
  for(int i=0; i<lockedChunks.size(); i++) {
    if(lockedChunks[i] == index) return true;
  }
  return false;
}



/** locking of the chunks is blocking and is strictly in ascending order.
    will only return if it gets the locks
    the set of nodes or elems that a chunk can ask to be locked can either be 
    shared, ghosts or local nodes/elements.
*/
int FEM_lock::lock(int numNodes, int *nodes, int numElems, int* elems, int elemType) {
  bool done = false;
  int ret = 0;
  CkAssert(!hasLocks && (lockedChunks.size()==0) && !isLocking && !isUnlocking); //should not try to lock while it has locks
  while(!done) {
    if((!isLocked || (isLocked && isOwner)) && !isUnlocking && !isLocking) {
      isLocking = true; //I've started modifying locked chunks, so I should be sure
      for(int i=0; i<numNodes; i++) {
	if(i==0) { //lock myself, just once
	  if(!existsChunk(idx)) {
	    lockedChunks.push_back(idx);
	  }
	}
	//which chunk does this belong to
	//add that chunk to the lock list, if it does not exist already.
	if(nodes[i] != -1) {
	  if(nodes[i] < -1) {
	    if(!mmod->fmMesh->node.ghost->is_valid(FEM_To_ghost_index(nodes[i]))) {
	      //clean up whatever has been done
	      lockedChunks.removeAll();
	      isLocking = false;
	      return 0;
	    }
	  } 
	  else if(!mmod->fmMesh->node.is_valid(nodes[i])) {
	    //clean up whatever has been done
	    lockedChunks.removeAll();
	    isLocking = false;
	    return 0;
	  }
	  int numchunks;
	  IDXL_Share **chunks1;
	  mmod->fmUtil->getChunkNos(0,nodes[i],&numchunks,&chunks1);
	  for(int j=0; j<numchunks; j++) {
	    if(!existsChunk(chunks1[j]->chk)) {
	      lockedChunks.push_back(chunks1[j]->chk);
	    }
	  }
	  for(int j=0; j<numchunks; j++) {
	    delete chunks1[j];
	  }
	  if(numchunks!=0) free(chunks1);
	}
      }
      for(int i=0; i<numElems; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if not already in it.
	if(elems[i] != -1) {
	  if(elems[i] < -1) {
	    if(!mmod->fmMesh->elem[elemType].ghost->is_valid(FEM_To_ghost_index(elems[i]))) {
	      //clean up whatever has been done
	      lockedChunks.removeAll();
	      isLocking = false;
	      return 0;
	    }
	  } 
	  else if(!mmod->fmMesh->elem[elemType].is_valid(elems[i])) {
	    //clean up whatever has been done
	    lockedChunks.removeAll();
	    isLocking = false;
	    return 0;
	  }
	  int numchunks;
	  IDXL_Share **chunks1;
	  mmod->fmUtil->getChunkNos(1,elems[i],&numchunks,&chunks1,elemType);
	  for(int j=0; j<numchunks; j++) {
	    if(!existsChunk(chunks1[j]->chk)) {
	      lockedChunks.push_back(chunks1[j]->chk);
	    }
	  }
	  for(int j=0; j<numchunks; j++) {
	    delete chunks1[j];
	  }
	  if(numchunks!=0) free(chunks1);
	}
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
	CkAssert(ret == 1);
	if(ret != 1) {
	  return -1;
	} else {
	  hasLocks = true;
	}
      }
      done = true;
      isLocking = false;
    }
    else {
      CthYield();
      //block
    }
  }
  CkAssert(!isLocking && !isUnlocking && hasLocks && lockedChunks.size()>0);
  return 1;
}

/** Since at one point of time one chunk can only lock one set of chunks for
    one operation, one does not need to pass arguments to unlock.
*/
int FEM_lock::unlock() {
  bool done = false;
  int ret = 0;
  if(!hasLocks && lockedChunks.size()==0) {
    return 0;
  }
  CkAssert(hasLocks && (lockedChunks.size()>0) && !isUnlocking && !isLocking); //should not try to unlock if it does not have locks
  while(!done) {
    if((!isLocked || (isLocked && isOwner)) && !isLocking && !isUnlocking) {
      //get rid of the locks
      isUnlocking = true;
      if(hasLocks) {
	for(int i=0; i<lockedChunks.size(); i++) {
	  ret = unlock(lockedChunks[i],idx);
	  CkAssert(ret == 1);
	  if(ret != 1) return -1;
	}
      }
      hasLocks = false;
      lockedChunks.removeAll(); //free up the list.. the next lock will build it again
      done = true;
      isUnlocking = false;
    }
    else {
      CthYield();
      //block
    }
  }
  CkAssert(!isUnlocking && !isLocking && !hasLocks && (lockedChunks.size()==0));
  return 1;
}

int FEM_lock::lock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  CkAssert(!(isLocked && (own==owner) && (chunkNo==idx))); //should not try to lock me, if it has already locked me
  while(true) {
    if((!isLocked || (chunkNo != idx)) && !isUnlocking) {
      if(chunkNo == idx) {
	isLocked = true;
	owner = own;
	if(owner == idx) {
	  isOwner = true;
	  hasLocks = true;
	}
	else {
	  isOwner = false;
	}
	CkPrintf("Chunk %d locked by chunk %d\n",chunkNo, own);
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	CkAssert(ret->i == 1);
	if(ret->i != 1) return -1;
	else {
	  hasLocks = true;
	}
      }
      break;
    }
    else {
      CthYield();
    }
  }
  delete ret;
  return 1;
}

/** For sanity, only the owner should unlock it
 */
int FEM_lock::unlock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  CkAssert(!(!isLocked && (chunkNo==idx))); //noone should try to unlock me, if I am not locked
  CkAssert(!(isLocked && (chunkNo==idx) && (owner!=own))); //if I am locked by someone else, only he should try to unlock me
  while(true) {
    if(isLocked || (chunkNo != idx) && !isLocking) {
      if(chunkNo == idx) {
	isLocked = false;
	owner = -1;
	isOwner = false;
	CkPrintf("Chunk %d unlocked by chunk %d\n",chunkNo, own);
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].unlockRemoteChunk(imsg);
	CkAssert(ret->i == 1);
	if(ret->i != 1) return -1;
      }
      break;
    }
    else {
      CthYield();
    }
  }
  delete ret;
  return 1;
}

