/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "ckgraph.h"
#include "CentralLB.h"
#include "DisableCoreLB.decl.h"

void CreateDisableCoreLB();
BaseLB * AllocateDisableCoreLB();

class DisableCoreLB : public CBase_DisableCoreLB {

public:
  struct HeapData {
    double load;
    int    pe;
    int    id;
  };

  DisableCoreLB(const CkLBOptions &);
  DisableCoreLB(CkMigrateMessage *m):CBase_DisableCoreLB(m) { lbname = "DisableCoreLB"; }
  void work(LDStats* stats);
private:
#if 1
  class ProcLoadGreater;
//  class ObjLoadGreater;

  enum           HeapCmp {GT = '>', LT = '<'};
      void           Heapify(HeapData*, int, int, HeapCmp);
  void           HeapSort(HeapData*, int, HeapCmp);
  void           BuildHeap(HeapData*, int, HeapCmp);
  bool        Compare(double, double, HeapCmp);
  HeapData*      BuildCpuArray(BaseLB::LDStats*, int, int *);
  HeapData*      BuildObjectArray(BaseLB::LDStats*, int, int *);
  bool        QueryBalanceNow(int step) override { return true; }
#endif
};

#endif /* _HEAPCENTLB_H_ */

/*@}*/
