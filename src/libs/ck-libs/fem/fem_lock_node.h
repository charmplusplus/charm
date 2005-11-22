/* File: fem_lock_node.h
 * Authors: Nilesh Choudhury
 * 
 */


#ifndef __CHARM_FEM_LOCK_H
#define __CHARM_FEM_LOCK_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"

class femMeshModify;

//there is one fem_lock associated with every node (locks on elements are not required)
//should lock all nodes, involved in any operation
class FEM_lockN {
  int owner, pending;
  femMeshModify *theMod;
  int idx; //index of the node
  int noreadLocks;
  int nowriteLocks;
  
 public:
  FEM_lockN() {};
  FEM_lockN(int i,femMeshModify *mod);
  ~FEM_lockN();

  void reset(int i,femMeshModify *mod);
  int rlock();
  int runlock();
  int wlock(int own);
  int wunlock(int own);
  bool haslocks();
  bool verifyLock(void);
  int lockOwner();
  int getIdx() { return idx; }
};

#endif
