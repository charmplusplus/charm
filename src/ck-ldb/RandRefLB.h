/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _RANDREFLB_H_
#define _RANDREFLB_H_

#include "CentralLB.h"
#include "RandRefLB.decl.h"

void CreateRandRefLB();

class RandRefLB : public CentralLB {
public:
  RandRefLB();
  RandRefLB(CkMigrateMessage *m) {}
private:
  CmiBool QueryBalanceNow(int step);
  LBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _RANDCENTLB_H_ */


/*@}*/
