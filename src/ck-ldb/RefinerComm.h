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
  struct CommTable {
    int* msgSentCount; // # of messages sent by each PE
    int* msgRecvCount; // # of messages received by each PE
    int* byteSentCount;// # of bytes sent by each PE
    int* byteRecvCount;// # of bytes reeived by each PE
    int count;
    CommTable(int p);
    ~CommTable();
    void clear();
    void increase(bool issend, int pe, int msgs, int bytes);
    double overheadOnPe(int pe);
  };
  CentralLB::LDStats* stats;
  CommTable *commTable;

  void create(int count, CentralLB::LDStats* , int* cur_p);
  void processorCommCost();
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  void commCost(int c, int pe, int &byteSent, int &msgSent, int &byteRecv, int &msgRecv);
  int refine();
  void computeAverageWithComm();
  double commAffinity(int c, int pe);
};

#endif /* _REFINERCOMM_H_ */


/*@}*/
