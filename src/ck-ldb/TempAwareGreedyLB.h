/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "CentralLB.h"
#include "TempAwareGreedyLB.decl.h"

void CreateTempAwareGreedyLB();
BaseLB * AllocateTempAwareGreedyLB();

class TempAwareGreedyLB : public CBase_TempAwareGreedyLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  TempAwareGreedyLB(const CkLBOptions &);
  TempAwareGreedyLB(CkMigrateMessage *m) : CBase_TempAwareGreedyLB(m) { lbname = "GreedyLB"; }
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
