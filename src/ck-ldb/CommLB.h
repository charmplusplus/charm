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

#ifndef _COMMLB_H_
#define _COMMLB_H_

#include "CentralLB.h"
#include "CommLB.decl.h"

#include "CommLBHeap.h"
#define CUT_OFF_FACTOR 1.200

struct obj_id{
  LDObjid oid;
  LDOMid mid;
};

struct graph{
  int id;
  int data;
  int nmsg;
  struct graph * next;
};

void CreateCommLB();

class CommLB : public CentralLB {
public:
  int nobj,npe;
  double ** alloc_array;
  graph * object_graph;
  obj_id * translate;
  int * htable;
  CommLB();
  CommLB(CkMigrateMessage *m);
private:
  CmiBool QueryBalanceNow(int step);
  LBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);
  void alloc(int pe, int id, double load);
  double compute_com(int id,int pe); 
  int search(LDObjid oid, LDOMid mid);
  void add_graph(int x, int y, int data, int nmsg);
  void make_hash();
};

#endif



/*@}*/



