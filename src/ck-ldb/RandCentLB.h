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

#ifndef _RANDCENTLB_H_
#define _RANDCENTLB_H_

#include "CentralLB.h"
#include "RandCentLB.decl.h"

void CreateRandCentLB();

class RandCentLB : public CentralLB {
public:
  RandCentLB();
  RandCentLB(CkMigrateMessage *m):CentralLB(m) {}
  void work(CentralLB::LDStats* stats, int count);
private:
  CmiBool QueryBalanceNow(int step);
};

#endif /* _RANDCENTLB_H_ */

/*@}*/
