#ifndef _HEAPCENTLB_H_
#define _HEAPCENTLB_H_

#include "CentralLB.h"
#include "HeapCentLB.decl.h"

void CreateHeapCentLB();

class HeapCentLB : public CentralLB {

struct HeapData {
	float cpuTime;
	int   pe;
	int   id;
};

public:
  HeapCentLB();
private:
  void    Heapify(HeapData *, int, int);
  CmiBool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
};

#endif /* _HEAPCENTLB_H_ */
