/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYCOMMLB_H_
#define _GREEDYCOMMLB_H_

#include "CentralLB.h"
#include "GreedyCommLB.decl.h"
#include "CommLBHeap.h"

#define CUT_OFF_FACTOR 1.200

void CreateGreedyCommLB();
BaseLB * AllocateGreedyCommLB();

struct graph{
  int id;
  int data;
  int nmsg;
  struct graph * next;
};

class processorInfo;

class GreedyCommLB : public CBase_GreedyCommLB {
public:
  int nobj,npe, nmigobj;
  int * assigned_array;
  processorInfo* processors;
  graph * object_graph;
  GreedyCommLB(const CkLBOptions &);
  GreedyCommLB(CkMigrateMessage *m);
  void pup(PUP::er &p){  }
  bool QueryBalanceNow(int step);
  void work(LDStats* stats);

private:
  void init();
  void alloc(int pe, int id, double load);
  double compute_com(LDStats* stats, int id,int pe);
  void add_graph(int x, int y, int data, int nmsg);
  void update(LDStats* stats, int id, int pe);

  double alpha, beta;
};

#endif


/*@}*/

