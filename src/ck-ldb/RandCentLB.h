#ifndef _RANDCENTLB_H_
#define _RANDCENTLB_H_

#include "CentralLB.h"
#include "RandCentLB.decl.h"

void CreateRandCentLB();

class RandCentLB : public CentralLB {
public:
  RandCentLB();
private:
  bool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _RANDCENTLB_H_ */
