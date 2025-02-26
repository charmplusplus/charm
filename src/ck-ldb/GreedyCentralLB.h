/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GreedyCentralLB_H_
#define _GreedyCentralLB_H_

#include "CentralLB.h"
#include "GreedyCentralLB.decl.h"

void CreateGreedyCentralLB();
BaseLB * AllocateGreedyCentralLB();

class GreedyCentralLB : public CBase_GreedyCentralLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  GreedyCentralLB(const CkLBOptions &);
  GreedyCentralLB(CkMigrateMessage *m):CBase_GreedyCentralLB(m) { lbname = "GreedyCentralLB"; }
  void work(LDStats* stats);
private:
  class ProcLoadGreater;
  class ObjLoadGreater;

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