/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _COMMAWARELB_H_
#define _COMMAWARELB_H_

#include "CentralLB.h"
#include "CommAwareRefineLB.decl.h"

void CreateCommAwareRefineLB();
BaseLB * AllocateCommAwareRefineLB();

class CommAwareRefineLB : public CBase_CommAwareRefineLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;

  };

  CommAwareRefineLB(const CkLBOptions &);
  CommAwareRefineLB(CkMigrateMessage *m):CBase_CommAwareRefineLB(m) {
    lbname = "CommAwareRefineLB";
  }
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

#endif /* _COMMAWARELB_H_ */

/*@}*/
