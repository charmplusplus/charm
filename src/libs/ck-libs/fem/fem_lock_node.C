#include "fem_lock_node.h"
#include "fem_mesh_modify.h"

FEM_lockN::FEM_lockN(int i) {
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
    CkPrintf("Got read lock on node %d\n", idx);
    noreadLocks++;
    return 1;
  }
  return -1; //should not reach here
}

int FEM_lockN::runlock() {
  CkAssert(noreadLocks>0 && nowriteLocks==0);
  if(noreadLocks > 0) {
    CkPrintf("Unlocked read lock on node %d\n", idx);
    noreadLocks--;
    return 1;
  }
  else {
    return -1;
  }
  return -1; //should not reach here
}

int FEM_lockN::wlock() {
  if(nowriteLocks==0 && noreadLocks==0) {
    nowriteLocks++;
    CkPrintf("Got write lock on node %d\n", idx);
    return 1;
  } else {
    return -1;
  }
  return -1;
}

int FEM_lockN::wunlock() {
  CkAssert(noreadLocks==0 && nowriteLocks>0);
  if(nowriteLocks) {
    CkPrintf("Unlocked write lock on node %d\n", idx);
    nowriteLocks--;
    return 1;
  } else {
    return -1;
  }
  return -1;
}
