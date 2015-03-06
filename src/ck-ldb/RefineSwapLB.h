/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "CentralLB.h"
#include "RefineSwapLB.decl.h"

void CreateRefineSwapLB();
BaseLB * AllocateRefineSwapLB();

class RefineSwapLB : public CBase_RefineSwapLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  RefineSwapLB(const CkLBOptions &);
  RefineSwapLB(CkMigrateMessage *m):CBase_RefineSwapLB(m) { lbname = "RefineSwapLB"; }
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
