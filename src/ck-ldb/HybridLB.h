/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef HYBRIDLB_H
#define HYBRIDLB_H

#include "CentralLB.h"
#include "HybridBaseLB.h"
#include "HybridLB.decl.h"

#include "topology.h"

void CreateHybridLB();

class HybridLB : public CBase_HybridLB
{
public:
  HybridLB(const CkLBOptions &);
  HybridLB(CkMigrateMessage *m): CBase_HybridLB(m) {}
  ~HybridLB();

protected:
  CentralLB *greedy;
  CentralLB *refine;

  virtual bool QueryBalanceNow(int) { return true; };  
  virtual bool QueryMigrateStep(int) { return true; };  
  virtual void work(LDStats* stats);

};

#endif /* NBORBASELB_H */

/*@}*/
