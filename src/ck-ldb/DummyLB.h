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

#ifndef _DUMMYLB_H_
#define _DUMMYLB_H_

#include "CentralLB.h"
#include "DummyLB.decl.h"

void CreateDummyLB();

class DummyLB : public CentralLB {
public:
  DummyLB();
  DummyLB(CkMigrateMessage *m):CentralLB(m) {}
private:
  CmiBool QueryBalanceNow(int step);
  LBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _DUMMYLB_H_ */

/*@}*/
