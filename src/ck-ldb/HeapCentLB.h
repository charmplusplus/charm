/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _HEAPCENTLB_H_
#define _HEAPCENTLB_H_

#include "CentralLB.h"
#include "HeapCentLB.decl.h"

void CreateHeapCentLB();

class HeapCentLB : public CentralLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  HeapCentLB();
  HeapCentLB(CkMigrateMessage *m) {}
private:
	enum           HeapCmp {GT = '>', LT = '<'};
    void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	CmiBool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(CentralLB::LDStats*, int, int *);      
	HeapData*      BuildObjectArray(CentralLB::LDStats*, int, int *);      
	CmiBool        QueryBalanceNow(int step);
	CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _HEAPCENTLB_H_ */
