#ifndef _RANDCENTLB_H_
#define _RANDCENTLB_H_

#include "CentralLB.h"
#include "MetisLB.decl.h"

void CreateMetisLB();

class MetisLB : public CentralLB {
public:
  MetisLB();
private:
  CmiBool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _RANDCENTLB_H_ */
