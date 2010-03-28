/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _REFINELB_H_
#define _REFINELB_H_

#include "CentralLB.h"
#include "Refiner.h"
#include "RefineLB.decl.h"

class minheap;
class maxheap;

#include "elements.h"
#include "ckheap.h"

void CreateRefineLB();
BaseLB *AllocateRefineLB();

class RefineLB : public CentralLB {
protected:
  computeInfo *computes;
  processorInfo *processors;
  minHeap *pes;
  maxHeap *computesHeap;
  int P;
  int numComputes;
  double averageLoad;

  double overLoad;

public:
  RefineLB(const CkLBOptions &);
  RefineLB(CkMigrateMessage *m):CentralLB(m) { lbname = (char *)"RefineLB"; }
  void work(BaseLB::LDStats* stats, int count);
private:
  CmiBool QueryBalanceNow(int step) { return CmiTrue; }

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

/*@}*/
