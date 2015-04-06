/**
 * Author Harshitha Menon (gplkrsh2@illinois.edu)
*/

#ifndef NODE_LEVEL_LB_H
#define NODE_LEVEL_LB_H

#include "CentralLB.h"
#include "HybridBaseLB.h"
#include "NodeLevelLB.decl.h"

void CreateNodeLevelLB();

class NodeLevelLB : public CBase_NodeLevelLB {
  private:
    CkVec<CentralLB *>  clbs;
    int num_levels;

  protected:
    virtual bool QueryBalanceNow(int) { return true; };
    virtual void work(LDStats* stats);

  public:
    NodeLevelLB(const CkLBOptions &);
    NodeLevelLB(CkMigrateMessage *m):CBase_NodeLevelLB(m) {}
};

#endif /* NODE_LEVEL_LB_H */
