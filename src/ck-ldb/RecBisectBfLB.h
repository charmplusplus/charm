/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _RECBISECTBFLB_H_
#define _RECBISECTBFLB_H_

#include "CentralLB.h"
#include "RecBisectBfLB.decl.h"

#include "ObjGraph.h"
#include "graph.h"
#include "bitvecset.h"
#include "cklists.h"

typedef CkQ<int> IntQueue;

void CreateRecBisectBfLB();
BaseLB * AllocateRecBisectBfLB();

typedef struct {
  int size;
  int * nodeArray;
} PartitionRecord;


typedef struct {
  int next;
  int max;
  PartitionRecord * partitions;
} PartitionList;


class RecBisectBfLB : public CBase_RecBisectBfLB {
public:
  RecBisectBfLB(const CkLBOptions &);
  RecBisectBfLB(CkMigrateMessage *m) : CBase_RecBisectBfLB(m) {}
private:
  bool QueryBalanceNow(int step);
  void work(LDStats* stats);

  Graph * convertGraph(ObjGraph *og);
  
  void  partitionInTwo(Graph *g, int nodes[], int numNodes, 
		 int ** pp1, int *numP1, int **pp2, int *numP2, 
		 int ratio1, int ratio2);  
  int findNextUnassigned(int max, BV_Set * all, BV_Set * s1, BV_Set * s2);
  float addToQ(IntQueue * q, Graph *g, BV_Set * all, BV_Set * s1, BV_Set * s2);
  
  void  enqChildren(IntQueue * q, Graph *g, BV_Set * all, 
	      BV_Set * s1, BV_Set * s2, int node) ;
  
  void addPartition(PartitionList * partitions, int * nodes, int num) ;
  void printPartitions(PartitionList * partitions) ;
  void recursivePartition(int numParts, Graph *g, int nodes[], int numNodes, 
		     PartitionList *partitions);

};



#endif /* _RECBISECTBFLB_H_ */


/*@}*/
