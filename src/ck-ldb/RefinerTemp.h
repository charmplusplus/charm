/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _REFINERTEMP_H_
#define _REFINERTEMP_H_

#include "elements.h"
#include "ckheap.h"
#include "CentralLB.h"

class RefinerTemp {
public:
  RefinerTemp(double _overload)
    : overLoad(_overload), P(CkNumPes()), computes(0)
    { 
    processors=0; 
    procFreq = new int[P];
    procFreqNew = new int[P];
    for (int i = 0; i < P; ++i)
      procFreq[i] = procFreqNew[i] = 1;

    sumFreqs=0;
  }

  RefinerTemp(double _overload,int *p,int *pn,int i);
  ~RefinerTemp() {}
  int sumFreqs,*procFreq,*procFreqNew;
  double totalInst;
  static int* AllocProcs(int count, BaseLB::LDStats* stats);
  static void FreeProcs(int* bufs);
  void Refine(int count, BaseLB::LDStats* stats, int* cur_p, int* new_p);

protected:
  void create(int count, BaseLB::LDStats* stats, int* cur_p);
  virtual int refine();
  int multirefine();
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  virtual void computeAverage();
  double computeMax();
double computeMax(int*);
  bool isHeavy(processorInfo *p);
  bool isLight(processorInfo *p);
  void removeComputes();

  double overLoad;
  double averageLoad;
  int P;
  int numAvail;
  int numComputes;
  computeInfo* computes;
  processorInfo* processors;
};

#endif /* _REFINERTEMP_H_ */


/*@}*/
