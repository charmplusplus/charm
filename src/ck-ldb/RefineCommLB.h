/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _REFINECOMMLB_H_
#define _REFINECOMMLB_H_

#include "CentralLB.h"
#include "RefinerComm.h"
#include "RefineLB.h"
#include "RefineCommLB.decl.h"

class minheap;
class maxheap;

void CreateRefineCommLB();
BaseLB *AllocateRefineCommLB();

class RefineCommLB : public CBase_RefineCommLB {
public:
  RefineCommLB(const CkLBOptions &);
  RefineCommLB(CkMigrateMessage *m):CBase_RefineCommLB(m) {}
private:
  bool QueryBalanceNow(int step);
  void work(LDStats* stats);

protected:
};

#endif /* _REFINELB_H_ */

/*@}*/
