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

class AdaptiveLB : public CentralLB
{
public:
  AdaptiveLB(const CkLBOptions &);
  AdaptiveLB(CkMigrateMessage *m):CentralLB(m) {}

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual void work(LDStats* stats);
  void computeNonlocalComm(long long &nmsgs, long long &nbytes);

private:  
//  CProxy_CentralLB thisProxy;
  CentralLB *greedyLB;
  CentralLB *refineLB;
  CentralLB *metisLB;
};

#endif /* CENTRAL_ADAPTIVE_LB_H */

/*@}*/


