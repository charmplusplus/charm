/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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

#define WEIGHTED 1
#define MULTI_CONSTRAINT 2

#endif /* _METISLB_H_ */
