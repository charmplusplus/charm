/* File: fem_lock_node.C
 * Authors: Nilesh Choudhury
 * 
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"

//#define DEBUG_LOCKS

FEM_lockN::FEM_lockN(int i,femMeshModify *mod) {
  owner = -1;
  pending = -1;
  theMod = mod;
  idx = i;
  noreadLocks = 0;
  nowriteLocks = 0;
}

FEM_lockN::~FEM_lockN() {
  //before deleting it, ensure that it is not holding any locks
}

void FEM_lockN::pup(PUP::er &p) {
  p|owner;
  p|pending;
  p|idx;
  p|noreadLocks;
  p|nowriteLocks;
}

void FEM_lockN::setMeshModify(femMeshModify *mod) {
  theMod = mod;
}



void FEM_lockN::reset(int i,femMeshModify *mod) {
  //CkAssert(noreadLocks==0 && nowriteLocks==0);
  if(haslocks()) wunlock(idx);
  owner = -1;
  pending = -1;
  theMod = mod;
  idx = i;
  noreadLocks = 0;
  nowriteLocks = 0;
}



int FEM_lockN::rlock() {
  if(nowriteLocks>0) { //if someone has a write lock, do not give read locks
    return -1;
  }
  else {
#ifdef DEBUG_LOCKS
    CkPrintf("Got read lock on node %d\n", FEM_My_partition(), idx);
#endif
    noreadLocks++;
    return 1;
  }
  return -1; //should not reach here
}

int FEM_lockN::runlock() {
  CkAssert(noreadLocks>0 && nowriteLocks==0);
  if(noreadLocks > 0) {
#ifdef DEBUG_LOCKS
    CkPrintf("[%d] Unlocked read lock on node %d\n", FEM_My_partition(), idx);
#endif
    noreadLocks--;
    return 1;
  }
  else {
    return -1;
  }
  return -1; //will not reach here
}

int FEM_lockN::wlock(int own) {
  if(nowriteLocks==0 && noreadLocks==0) {
    nowriteLocks++;
    owner = own;
#ifdef DEBUG_LOCKS
    CkPrintf("[%d] Got write lock on node %d{%d} .\n",owner, idx, theMod->idx);
#endif
    CkAssert(nowriteLocks==1);
    if(pending==own) pending=-1; //got the lock, reset pending
    return 1;
  } else {
    if((pending!=-1 && own<pending) || (pending==-1 && own<owner)) {
      pending = own; //set pending as it has higher priority
      return -1; //keep trying
    }
    return -2; //give up trying for a while
    }
  return -2;
}

int FEM_lockN::wunlock(int own) {
  /*if(!(noreadLocks==0 && nowriteLocks>0)) {
    CkPrintf("[%d] Error:: unlocking unacquired write lock %d{%d} .\n",owner, idx, theMod->idx);
    nowriteLocks=1;
    }*/
  //CkAssert(noreadLocks==0 && nowriteLocks>0);
  if(nowriteLocks>0) {
    nowriteLocks--;
    CkAssert(nowriteLocks==0);
#ifdef DEBUG_LOCKS
    CkPrintf("[%d] Unlocked write lock on node %d{%d} .\n",owner, idx, theMod->idx);
#endif
    owner = -1;
    return 1;
  } else {
    return -1;
  }
  return -1;
}



bool FEM_lockN::haslocks() {
  if(noreadLocks>0 || nowriteLocks>0) {
    return true;
  }
  else return false;
}

bool FEM_lockN::verifyLock(void) {
  const IDXL_Rec *irec = theMod->fmMesh->node.shared.getRec(idx);
  if(irec) {
    int minchunk = theMod->idx;
    for(int i=0; i<irec->getShared(); i++) {
      int pchk = irec->getChk(i);
      if(pchk<minchunk) minchunk=pchk;
    }
    //if someone wants to lock me, I should be on the smallest chunk
    if(minchunk!=theMod->idx) return false;
  }
  if(nowriteLocks==1) return true;
  else return false;
}

int FEM_lockN::lockOwner() {
  if(noreadLocks>0 || nowriteLocks>0) {
    return owner;
  }
  else return -1;
}
