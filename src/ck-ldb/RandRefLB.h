#ifndef _RANDREFLB_H_
#define _RANDREFLB_H_

#include "CentralLB.h"
#include "RandRefLB.decl.h"

void CreateRandRefLB();

class RandRefLB : public CentralLB {
public:
  RandRefLB();
private:
  CmiBool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _RANDCENTLB_H_ */
