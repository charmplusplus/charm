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

#ifndef _REFINECOMMLB_H_
#define _REFINECOMMLB_H_

#include "CentralLB.h"
#include "RefinerComm.h"
#include "RefineLB.h"
#include "RefineCommLB.decl.h"

class minheap;
class maxheap;

#include "elements.h"
#include "ckheap.h"

void CreateRefineCommLB();
BaseLB *AllocateRefineCommLB();

class RefineCommLB : public RefineLB {
public:
  RefineCommLB(const CkLBOptions &);
  RefineCommLB(CkMigrateMessage *m):RefineLB(m) {}
private:
  CmiBool QueryBalanceNow(int step);
  void work(BaseLB::LDStats* stats, int count);

protected:
};

#endif /* _REFINELB_H_ */

/*@}*/
