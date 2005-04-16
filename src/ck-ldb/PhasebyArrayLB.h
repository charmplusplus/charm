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

#ifndef _CPAIMDLB_H_
#define _CPAIMDLB_H_

#include "CentralLB.h"
#include "PhasebyArrayLB.decl.h"

void CreatePhasebyArrayLB();

class PhasebyArrayLB : public CentralLB {
public:
  PhasebyArrayLB(const CkLBOptions &);
  PhasebyArrayLB(CkMigrateMessage *m):CentralLB(m) {}
private:
	//CkVec<CProxy_ArrayBase> arrayProxies;
  BaseLB::LDStats *tempStats;
	CentralLB *lb;
	CkVec<LDOMid> omids;
	CmiBool QueryBalanceNow(int step);
	void copyStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats);
	void updateStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats);
	void work(BaseLB::LDStats* stats, int count);
};

#endif /* _DUMMYLB_H_ */

/*@}*/
