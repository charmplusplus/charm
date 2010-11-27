/** \file MetisLB.h
 *
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _METISLB_H_
#define _METISLB_H_

#include "CentralLB.h"
#include "MetisLB.decl.h"

#define WEIGHTED 1
#define MULTI_CONSTRAINT 2

void CreateMetisLB();
BaseLB * AllocateMetisLB();

class MetisLB : public CentralLB {
public:
  MetisLB(const CkLBOptions &);
  MetisLB(CkMigrateMessage *m):CentralLB(m) { lbname = "MetisLB"; }
private:
  CmiBool QueryBalanceNow(int step) { return CmiTrue; }
  void work(LDStats* stats);
};

#endif /* _METISLB_H_ */

/*@}*/
