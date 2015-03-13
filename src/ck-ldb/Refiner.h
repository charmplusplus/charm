/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _REFINER_H_
#define _REFINER_H_

#include "elements.h"
#include "ckheap.h"
#include "CentralLB.h"

class Refiner {
public:
  Refiner(double _overload) { 
    overLoad = _overload; computes=0; processors=0; 
  };
  ~Refiner() {}

  static int* AllocProcs(int count, BaseLB::LDStats* stats);
  static void FreeProcs(int* bufs);
  void Refine(int count, BaseLB::LDStats* stats, int* cur_p, int* new_p);

  double computeAverageLoad();
  double computeMax();

protected:
  void create(int count, BaseLB::LDStats* stats, int* cur_p);
  virtual int refine();
  int multirefine(bool reset = 1);
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  virtual void computeAverage();
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
