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

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "CentralLB.h"
#include "GreedyLB.decl.h"

void CreateGreedyLB();

class GreedyLB : public CentralLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  GreedyLB();
  GreedyLB(CkMigrateMessage *m):CentralLB(m) {}
  void work(LDStats* stats,int count);
private:
	enum           HeapCmp {GT = '>', LT = '<'};
    	void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	CmiBool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(CentralLB::LDStats*, int, int *);  
	HeapData*      BuildObjectArray(CentralLB::LDStats*, int, int *);      
	CmiBool        QueryBalanceNow(int step);
};

#endif /* _HEAPCENTLB_H_ */

/*@}*/
