/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _GREEDYREFLB_H_
#define _GREEDYREFLB_H_

#include "CentralLB.h"
#include "Refiner.h"
#include "GreedyRefLB.decl.h"

void CreateGreedyRefLB();

class GreedyRefLB : public CentralLB {

struct HeapData {
	float load;
	int   pe;
	int   id;
};

public:
  GreedyRefLB();
private:
  void           Heapify(HeapData *, int, int);
  void           HeapSort(HeapData*, int);
	void           BuildHeap(HeapData*, int);
  HeapData*      BuildCpuArray(CentralLB::LDStats*, int, int *);      
  HeapData*      BuildObjectArray(CentralLB::LDStats*, int, int *);      
  CmiBool        QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _GREEDYREFLB_H_ */
