#ifndef _REFINELB_H_
#define _REFINELB_H_

#include "CentralLB.h"
#include "Refiner.h"
#include "RefineLB.decl.h"

class minheap;
class maxheap;

#include "elements.h"
#include "heap.h"

void CreateRefineLB();

class RefineLB : public CentralLB {
protected:
  computeInfo *computes;
  processorInfo *processors;
  minHeap *pes;
  maxHeap *computesHeap;
  int P;
  int numComputes;
  double averageLoad;

public:
  double overLoad;

public:
  RefineLB();
private:
  CmiBool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);

protected:
/*
  void create(LDStats* stats, int count);
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  void computeAverage();
  double computeMax();
  int refine();
*/
};

#endif /* _REFINELB_H_ */
