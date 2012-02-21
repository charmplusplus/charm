/** \file ZoltanLB.h
 *
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _ZOLTANLB_H_
#define _ZOLTANLB_H_

#include "CentralLB.h"
#include "ZoltanLB.decl.h"

#define WEIGHTED 1
#define MULTI_CONSTRAINT 2

void CreateZoltanLB();
BaseLB * AllocateZoltanLB();

class ZoltanLB : public CentralLB {
public:
  ZoltanLB(const CkLBOptions &);
  ZoltanLB(CkMigrateMessage *m):CentralLB(m) { lbname = "ZoltanLB"; }
private:
  CmiBool QueryBalanceNow(int step) { return CmiTrue; }
  void work(LDStats* stats);
};

#endif /* _ZOLTANLB_H_ */

/*@}*/
