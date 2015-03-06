/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _CommLB_H_
#define _CommLB_H_

#include <CentralLB.h>
#include "CommLB.decl.h"

#include "CommLBHeap.h"
#include "GreedyCommLB.h"

#define CUT_OFF_FACTOR 1.200

void CreateCommLB();

struct alloc_struct{
  double load;
  int nbyte;
  int nmsg;
};

class CommLB : public CBase_CommLB {
public:
  int nobj,npe;
  alloc_struct ** alloc_array;
  graph * object_graph;
  CommLB(const CkLBOptions &);
  CommLB(CkMigrateMessage *m):CBase_CommLB(m) {}
private:
  bool QueryBalanceNow(int step);
  void work(LDStats* stats);
  void alloc(int pe, int id, double load, int nmsg, int nbyte);
  double compute_cost(int id, int pe, int n_alloc, int &out_msg, int &out_byte); 
  void add_graph(int x, int y, int data, int nmsg);
};

#endif


/*@}*/

