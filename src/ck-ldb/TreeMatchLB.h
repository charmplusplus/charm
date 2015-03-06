/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _TREEMATCHLB_H_
#define _TREEMATCHLB_H_

#include "CentralLB.h"
#include "TreeMatchLB.decl.h"

void CreateTreeMatchLB();

class TreeMatchLB : public CBase_TreeMatchLB {
public:
  TreeMatchLB(const CkLBOptions &);
  TreeMatchLB(CkMigrateMessage *m):CBase_TreeMatchLB(m) {}
  void work(BaseLB::LDStats* stats);
private:
  bool QueryBalanceNow(int step);
};

#endif /* _TREEMATCHLB_H_ */

/*@}*/
