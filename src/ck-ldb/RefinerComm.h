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

#ifndef _REFINERCOMM_H_
#define _REFINERCOMM_H_

#include "CentralLB.h"

#include "elements.h"
#include "heap.h"
#include "Refiner.h"

class RefinerComm : public Refiner {
public:
  RefinerComm(double _overload): Refiner(_overload)  { 
    overLoad = _overload; computes=0; processors=0; 
  };
  ~RefinerComm() {}

  void Refine(int count, CentralLB::LDStats* stats, int* cur_p, int* new_p);

private:
  int* msgSentCount; // # of messages sent by each PE
  int* msgRecvCount; // # of messages received by each PE
  int* byteSentCount;// # of bytes sent by each PE
  int* byteRecvCount;// # of bytes reeived by each PE
  void create(int count, CentralLB::LDStats* stats, int* cur_p);
  void commSummary(CentralLB::LDStats* stats);
  double commCost(int);
  int refine();
  void computeAverage();
};

#endif /* _REFINERCOMM_H_ */


/*@}*/
