
#ifndef _RECBISECTBFLB_H_
#define _RECBISECTBFLB_H_

#include "CentralLB.h"
#include "RecBisectBfLB.decl.h"

#include "ObjGraph.h"
#include "graph.h"
#include "bitvecset.h"
#include "fifoInt.h"

void CreateRecBisectBfLB();

typedef struct {
  int size;
  int * nodeArray;
} PartitionRecord;


typedef struct {
  int next;
  int max;
  PartitionRecord * partitions;
} PartitionList;


class RecBisectBfLB : public CentralLB {
public:
  RecBisectBfLB();
private:
  CmiBool QueryBalanceNow(int step);
  CLBMigrateMsg* Strategy(CentralLB::LDStats* stats, int count);

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
