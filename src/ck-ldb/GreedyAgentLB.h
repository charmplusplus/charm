/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYAGENTLB_H_
#define _GREEDYAGENTLB_H_

#include "CentralLB.h"
#include "GreedyAgentLB.decl.h"
#include "LBAgent.h"

void CreateGreedyAgentLB();

class GreedyAgentLB : public CBase_GreedyAgentLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  GreedyAgentLB(const CkLBOptions &);
  GreedyAgentLB(CkMigrateMessage *m):CBase_GreedyAgentLB(m) { lbname = "GreedyAgentLB"; }
  void work(LDStats* stats);
private:
	TopologyAgent				*topologyAgent;
	enum           HeapCmp {GT = '>', LT = '<'};
  void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	bool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(CentralLB::LDStats*, int, int *);  
	HeapData*      BuildObjectArray(CentralLB::LDStats*, int, int *);      
	bool        QueryBalanceNow(int step);
};

#endif /* _GREEDYAGENTLB_H_ */

/*@}*/
