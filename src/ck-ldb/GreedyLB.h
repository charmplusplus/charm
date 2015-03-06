/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "CentralLB.h"
#include "GreedyLB.decl.h"

void CreateGreedyLB();
BaseLB * AllocateGreedyLB();

class GreedyLB : public CBase_GreedyLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  GreedyLB(const CkLBOptions &);
  GreedyLB(CkMigrateMessage *m):CBase_GreedyLB(m) { lbname = "GreedyLB"; }
  void work(LDStats* stats);
private:
	enum           HeapCmp {GT = '>', LT = '<'};
    	void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	bool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(BaseLB::LDStats*, int, int *);  
	HeapData*      BuildObjectArray(BaseLB::LDStats*, int, int *);      
	bool        QueryBalanceNow(int step);
};

#endif /* _HEAPCENTLB_H_ */

/*@}*/
