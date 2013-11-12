/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _REFINERCOMM_H_
#define _REFINERCOMM_H_

#include "CentralLB.h"
#include "Refiner.h"

class RefinerComm : public Refiner {
public:
  RefinerComm(double _overload): Refiner(_overload)  { 
    overLoad = _overload; computes=0; processors=0; 
  };
  ~RefinerComm() {}

  void Refine(int count, BaseLB::LDStats* stats, int* cur_p, int* new_p);

private:
  struct Messages {
    int byteSent;
    int msgSent;
    int byteRecv; 
    int msgRecv;
    Messages() { clear(); }
    void clear() { byteSent=msgSent=byteRecv=msgRecv=0; }
    double cost() {
      return msgSent * _lb_args.alpha() + 
             byteSent * _lb_args.beta() +
             msgRecv * PER_MESSAGE_RECV_OVERHEAD + 
             byteRecv * PER_BYTE_RECV_OVERHEAD;
    }
  };
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
  BaseLB::LDStats* stats;
  CommTable *commTable;

  void create(int count, BaseLB::LDStats* , int* cur_p);
  void processorCommCost();
  void assign(computeInfo *c, int p);
  void assign(computeInfo *c, processorInfo *p);
  void deAssign(computeInfo *c, processorInfo *pRec);
  virtual int refine();
  virtual void computeAverage();
  void objCommCost(int c, int pe, Messages &m);
  void commAffinity(int c, int pe, Messages &m);
  inline void printLoad() {
      for (int i=0; i<P; i++) CmiPrintf("%f ", processors[i].load);
      CmiPrintf("\n");
  }
};

#endif /* _REFINERCOMM_H_ */


/*@}*/
