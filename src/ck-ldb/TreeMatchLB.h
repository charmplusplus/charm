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

#ifndef _TREEMATCHLB_H_
#define _TREEMATCHLB_H_

#include "CentralLB.h"
#include "TreeMatchLB.decl.h"

void CreateTreeMatchLB();

class TreeMatchLB : public CentralLB {
public:
  TreeMatchLB(const CkLBOptions &);
  TreeMatchLB(CkMigrateMessage *m):CentralLB(m) {}
  void work(BaseLB::LDStats* stats);
private:
  CmiBool QueryBalanceNow(int step);
};

#endif /* _TREEMATCHLB_H_ */

/*@}*/
