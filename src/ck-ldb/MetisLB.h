#ifndef _METISLB_H_
#define _METISLB_H_

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

#endif /* _METISLB_H_ */
