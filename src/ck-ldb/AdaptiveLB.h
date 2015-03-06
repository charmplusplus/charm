/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef CENTRAL_ADAPTIVE_LB_H
#define CENTRAL_ADAPTIVE_LB_H

#include "CentralLB.h"
#include "AdaptiveLB.decl.h"

void CreateAdaptiveLB();

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class AdaptiveLB : public CBase_AdaptiveLB
{
public:
  AdaptiveLB(const CkLBOptions &);
  AdaptiveLB(CkMigrateMessage *m):CBase_AdaptiveLB(m) {}

protected:
  virtual bool QueryBalanceNow(int) { return true; };  
  virtual void work(LDStats* stats);
//  void computeNonlocalComm(long long &nmsgs, long long &nbytes);

private:  
//  CProxy_CentralLB thisProxy;
  CentralLB *greedyLB;
  CentralLB *refineLB;
  CentralLB *metisLB;
  CentralLB *commRefineLB;
  MetaBalancer* metabalancer;
};

#endif /* CENTRAL_ADAPTIVE_LB_H */

/*@}*/


