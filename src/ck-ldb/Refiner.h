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

#ifndef _REFINER_H_
#define _REFINER_H_

#include "CentralLB.h"

class minheap;
class maxheap;

#include "elements.h"
#include "heap.h"

class Refiner {
public:
  Refiner(double _overload) { 
    overLoad = _overload; computes=0; processors=0; 
  };
  ~Refiner() { delete [] computes; delete [] processors; };

  static int** AllocProcs(int count, CentralLB::LDStats* stats);
  static void FreeProcs(int** bufs);
  void Refine(int count, CentralLB::LDStats* stats, int** cur_p, int** new_p);

private:
  void create(int count, CentralLB::LDStats* stats, int** cur_p);
  int refine();
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  void computeAverage();
  double computeMax();
  int isHeavy(processorInfo *p);
  int isLight(processorInfo *p);
  void removeComputes();

  double overLoad;
  double averageLoad;
  int P;
  int numAvail;
  int numComputes;
  computeInfo* computes;
  processorInfo* processors;
};

#endif /* _REFINER_H_ */


/*@}*/
