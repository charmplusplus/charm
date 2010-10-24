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

#ifndef _PHASEBYARRAYLB_H_
#define _PHASEBYARRAYLB_H_

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
	CkVec<CmiBool> migratableOMs;
	CmiBool QueryBalanceNow(int step);
	void copyStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats);
	void updateStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats);
	void work(LDStats* stats);
};

#endif /* _PHASEBYARRAYLB_H_ */

/*@}*/
