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

#ifndef _GREEDYREFLB_H_
#define _GREEDYREFLB_H_

#include "CentralLB.h"
#include "Refiner.h"
#include "GreedyRefLB.decl.h"

void CreateGreedyRefLB();

class GreedyRefLB : public CentralLB {

struct HeapData {
	double load;
	int   pe;
	int   id;
};

public:
  GreedyRefLB();
  GreedyRefLB(CkMigrateMessage *m) {}
private:
	enum           GreedyCmp {GT = '>', LT = '<'};
	CmiBool        Compare(double, double, GreedyCmp);
    void           Heapify(HeapData *, int, int, GreedyCmp);
	void           HeapSort(HeapData*, int, GreedyCmp);
	void           BuildHeap(HeapData*, int, GreedyCmp);
	HeapData*      BuildCpuArray(CentralLB::LDStats*, int, int *);      
	HeapData*      BuildObjectArray(CentralLB::LDStats*, int, int *);      
	CmiBool        QueryBalanceNow(int step);
	LBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _GREEDYREFLB_H_ */

/*@}*/
