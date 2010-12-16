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
  DummyLB(const CkLBOptions &);
  DummyLB(CkMigrateMessage *m):CentralLB(m) {}
private:
  CmiBool QueryBalanceNow(int step);
  void work(LDStats* stats) {}
};

#endif /* _DUMMYLB_H_ */

/*@}*/
