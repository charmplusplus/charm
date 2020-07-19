/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _DUMMYLB_H_
#define _DUMMYLB_H_

#include "CentralLB.h"
#include "DummyLB.decl.h"

void CreateDummyLB();

class DummyLB : public CBase_DummyLB {
public:
  DummyLB(const CkLBOptions &);
  DummyLB(CkMigrateMessage *m):CBase_DummyLB(m) {}
private:
  bool QueryBalanceNow(int step);
  void work(LDStats* stats) {}
};

#endif /* _DUMMYLB_H_ */

/*@}*/
