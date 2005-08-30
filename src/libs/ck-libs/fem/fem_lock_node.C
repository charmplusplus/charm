#include "fem_lock_node.h"
#include "fem_mesh_modify.h"

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

int FEM_lockN::rlock() {
  if(nowriteLocks>0) { //if someone has a write lock, do not give read locks
    return -1;
  }
  else {
    //CkPrintf("Got read lock on node %d\n", idx);
    noreadLocks++;
    return 1;
  }
  return -1; //should not reach here
}

int FEM_lockN::runlock() {
  CkAssert(noreadLocks>0 && nowriteLocks==0);
  if(noreadLocks > 0) {
    //CkPrintf("Unlocked read lock on node %d\n", idx);
    noreadLocks--;
    return 1;
  }
  else {
    return -1;
  }
  return -1; //should not reach here
}

int FEM_lockN::wlock(int own) {
  if(nowriteLocks==0 && noreadLocks==0) {
    nowriteLocks++;
    owner = own;
#ifdef DEBUG_LOCKS
    CkPrintf("[%d] Got write lock on node %d{%d} .\n",owner, idx, theMod->idx);
#endif
    return 1;
  } else {
    return -1;
  }
  return -1;
}

int FEM_lockN::wunlock(int own) {
  CkAssert(noreadLocks==0 && nowriteLocks>0);
  if(nowriteLocks) {
    nowriteLocks--;
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
