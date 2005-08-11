#ifndef __CHARM_FEM_LOCK_H
#define __CHARM_FEM_LOCK_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"

#define _LOCKNODES

class femMeshModify;

//there is one fem_lock associated with every node (locks on elements are not required)
//should lock all nodes, involved in any operation
class FEM_lockN {
  int idx; //index of the node
  int noreadLocks;
  int nowriteLocks;
    
 public:
  FEM_lockN() {};
  FEM_lockN(int i);
  ~FEM_lockN();
    
  int rlock();
  int runlock();
  int wlock();
  int wunlock();
  int getIdx() { return idx; }
};

#endif
