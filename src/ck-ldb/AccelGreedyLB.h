/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _ACCELGREEDYLB_H_
#define _ACCELGREEDYLB_H_

#include "CentralLB.h"
#include "AccelGreedyLB.decl.h"

void CreateAccelGreedyLB();
BaseLB * AllocateAccelGreedyLB();

class AccelGreedyLB : public CentralLB {

 public:

  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  AccelGreedyLB(const CkLBOptions &);
  AccelGreedyLB(CkMigrateMessage *m):CentralLB(m) { lbname = "AccelGreedyLB"; }
  void work(LDStats* stats);

 private:

	enum           HeapCmp {GT = '>', LT = '<'};
	void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	bool           Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(BaseLB::LDStats*, int, int *);
	HeapData*      BuildObjectArray(BaseLB::LDStats*, int, int *);
	bool        QueryBalanceNow(int step);
};

#endif /* _HEAPCENTLB_H_ */

/*@}*/
