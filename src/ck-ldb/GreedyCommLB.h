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

#ifndef _GREEDYCOMMLB_H_
#define _GREEDYCOMMLB_H_

#include "CentralLB.h"
#include "GreedyCommLB.decl.h"

#include "CommLBHeap.h"
#define CUT_OFF_FACTOR 1.200

void CreateGreedyCommLB();

struct graph{
  int id;
  int data;
  int nmsg;
  struct graph * next;
};

class GreedyCommLB : public CentralLB {
public:
  int nobj,npe, nmigobj;
  double ** alloc_array;
  graph * object_graph;
  GreedyCommLB(const CkLBOptions &);
  GreedyCommLB(CkMigrateMessage *m);
private:
  CentralLB::LDStats* stats;
  CmiBool QueryBalanceNow(int step);
  void work(CentralLB::LDStats* stats, int count);
  void alloc(int pe, int id, double load);
  double compute_com(int id,int pe); 
  void add_graph(int x, int y, int data, int nmsg);
  void update(int id, int pe);
};

#endif



/*@}*/



