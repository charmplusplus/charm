#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"


#include "RecBisectBfLB.h"
#include "RecBisectBfLB.def.h"

extern "C" {
IntQueue * fifoInt_create(int size);
int fifoInt_enqueue(IntQueue *q, int value);
int fifoInt_empty(IntQueue *q);
int fifoInt_dequeue(IntQueue *q);
void fifoInt_destroy(IntQueue* q);
}

extern "C" {
  Graph * initGraph(int v, int e);
  void freeGraph(Graph* g);
  void nextVertex(Graph *g, int v, float w);
  void finishVertex(Graph *g);
  void addEdge(Graph *g, int w, float w2);
  float graph_weightof(Graph *g, int vertex);
  int numNeighbors(Graph *g, int node);
  int getNeighbor(Graph *g, int d , int i);
  void printGraph(Graph *g);

  int bvset_size(BV_Set *);
  int bvset_find(BV_Set *, int i);
  void bvset_enumerate(BV_Set * s1, int **p1, int *numP1);
  void bvset_insert(BV_Set *ss1, int t1);
  BV_Set *makeSet(int *nodes, int numNodes, int V);
  BV_Set *makeEmptySet( int V);
  void destroySet(BV_Set* s);
}



void CreateRecBisectBfLB()
{
  //  CkPrintf("[%d] creating RecBisectBfLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RecBisectBfLB::ckNew();
  //  CkPrintf("[%d] created RecBisectBfLB %d\n",CkMyPe(),loadbalancer);
}

RecBisectBfLB::RecBisectBfLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] RecBisectBfLB created\n",CkMyPe());
}

CmiBool RecBisectBfLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

CLBMigrateMsg* RecBisectBfLB::Strategy(CentralLB::LDStats* stats, 
				       int numPartitions)
{
  int i;
  PartitionList *partitions;

  
  CkPrintf("[%d] RecBisectBfLB strategy\n",CkMyPe());
  ObjGraph og(numPartitions, stats);

  Graph *g =  convertGraph( &og);
  CkPrintf("[%d] RecBisectBfLB: graph converted\n",CkMyPe());
    
  //  printGraph(g);
  int* nodes = (int *) malloc(sizeof(int)*g->V);

  for (i=0; i<g->V; i++)
    nodes[i] = i;

  partitions = (PartitionList *) malloc(sizeof(PartitionList));
  partitions->next = 0;
  partitions->max = numPartitions;
  partitions->partitions = (PartitionRecord *)
    malloc(sizeof(PartitionRecord)* numPartitions);
    
  recursivePartition(numPartitions, g, nodes, g->V,  partitions);
  
  //  CmiPrintf("\ngraph partitioned\n");

  freeGraph(g);
  
  //  printPartitions(partitions);

  CkVector migrateInfo;

  for (i=0; i<partitions->max; i++) {
    //    CmiPrintf("[%d] (%d) : \t", i, partitions->partitions[i].size);
    int j;
    for (j=0; j< partitions->partitions[i].size; j++) {
      //      CmiPrintf("%d ", partitions->partitions[i].nodeArray[j]);
      const int objref = partitions->partitions[i].nodeArray[j];
      ObjGraph::Node n = og.GraphNode(objref);
      /*     CkPrintf("Moving %d(%d) from %d to %d\n",objref,
	     stats[n.proc].objData[n.index].handle.id.id[0],n.proc,i);
      */
   
      if (n.proc != i) {
	MigrateInfo *migrateMe = new MigrateInfo;
	migrateMe->obj = stats[n.proc].objData[n.index].handle;
	migrateMe->from_pe = n.proc;
	migrateMe->to_pe = i;
	migrateInfo.push_back((void*)migrateMe);
      }
    }
    free(partitions->partitions[i].nodeArray);
  }
  free(partitions->partitions);
  free(partitions);

  int migrate_count=migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  CkPrintf("Moving %d elements\n",migrate_count);
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  CmiPrintf("returning from partitioner strategy\n");
  return msg;
};


Graph *
RecBisectBfLB::convertGraph(ObjGraph *og) {

  Graph *g;
  
  int V, E, i;

  V = og->NodeCount();
  E = og->EdgeCount();

  g = initGraph(V, E);

  //  CkPrintf("[%d] RecBisectBfLB: convert (v=%d, e=%d, g=%p\n",
  //	   CkMyPe(), V, E, g);

  for (i =0; i<V; i++) {
    nextVertex(g, i, og->LoadOf(i));

    ObjGraph::Node n = og->GraphNode(i);
    ObjGraph::Edge *l;

    //  CkPrintf("[%d] RecBisectBfLB: convert before addEdge Loop\n");
    
    l = n.edges_from();
    while (l) {
      // CkPrintf("[%d] RecBisectBfLB: convert in addEdge Loop1\n");
      addEdge(g, l->to_node, 1.0); /* get edgeweight */
      l = l->next_from();
    }

    l = n.edges_to();
    while (l) {
      // CkPrintf("[%d] RecBisectBfLB: convert in addEdge Loop2\n");
      addEdge(g, l->from_node, 1.0); /* get edgeweight */
      l = l->next_to();
    }
    finishVertex(g);
  }
  return g;
}

void RecBisectBfLB::partitionInTwo(Graph *g, int nodes[], int numNodes, 
	       int ** pp1, int *numP1, int **pp2, int *numP2, 
	       int ratio1, int ratio2)
{
  int r1, r2, weight1, weight2;
  BV_Set *all, *s1, *s2; 
  IntQueue * q1, *q2;
  int * p1, *p2;

  r1 = nodes[0];
  r2 = nodes[numNodes-1];
  /* Improvement:
     select r1 and r2 more carefully: 
     e.g. farthest away from each other. */

  all = makeSet(nodes, numNodes, g->V);
  s1 = makeEmptySet(g->V);
  s2 = makeEmptySet(g->V);

  q1 = fifoInt_create(g->V);
  q2 = fifoInt_create(g->V);

  fifoInt_enqueue(q1, r1);
  fifoInt_enqueue(q2, r2);
  /*  printf("r1=%d, r2=%d\n", r1, r2);*/

  weight1 = 0; weight2 = 0;
  while (   (bvset_size(s1) + bvset_size(s2)) < numNodes ) {
    if (weight1*ratio2 < weight2*ratio1) 
      weight1 += addToQ(q1, g, all, s1,s2);      
    else 
      weight2 += addToQ(q2, g, all, s2,s1);
  }

  bvset_enumerate(s1, &p1, numP1);
  bvset_enumerate(s2, &p2, numP2);
  *pp1 = p1;
  *pp2 = p2;
  destroySet(s1);
  destroySet(s1);
  fifoInt_destroy(q1);
  fifoInt_destroy(q2);
}

int 
RecBisectBfLB::findNextUnassigned(int max, BV_Set * all, 
				  BV_Set * s1, BV_Set * s2)
{
  int i;
  for (i=0; i<max; i++) { 
    if (bvset_find(all, i))
      if ( (!bvset_find(s1,i)) && (!bvset_find(s2,i)) ) 
	return i;
  }
  return (max + 1);
}

float RecBisectBfLB::addToQ(IntQueue * q, Graph *g, BV_Set * all, 
			    BV_Set * s1, BV_Set * s2)
{
  int t1, doneUpto;
  float weightAdded;

  weightAdded = 0.0;

  if (fifoInt_empty(q)) {
    doneUpto = findNextUnassigned(g->V, all, s1, s2);
    if (doneUpto < g->V) fifoInt_enqueue(q,doneUpto); 
  }
  if (!fifoInt_empty(q) ) {
    t1 = fifoInt_dequeue(q);
    if (bvset_find(all, t1)) /* t1 is a vertex of the given partition */
      if ( (!bvset_find(s1, t1)) && ( !bvset_find(s2, t1)) ) {
	bvset_insert(s1, t1);
	weightAdded = graph_weightof(g, t1); 
	/*	 printf("adding %d to s\n", t1); */
	enqChildren(q, g, all, s1, s2, t1);
      }
  }
  return weightAdded;
}

void RecBisectBfLB::enqChildren(IntQueue * q, Graph *g, BV_Set * all, 
				BV_Set * s1, BV_Set * s2, int node)
{
  int nbrs, i, j;

  nbrs = numNeighbors(g, node);
  for (i=0; i<nbrs; i++) {
    j = getNeighbor(g, node, i);
    if (  (bvset_find(all,j)) && (!bvset_find(s1,j)) 
	  && (!bvset_find(s2,j)) ) {
      fifoInt_enqueue(q, j);
    }
  } 
}


void RecBisectBfLB::addPartition(PartitionList * partitions, 
				 int * nodes, int num) 
{
  int i;
  i =  partitions->next++;
  partitions->partitions[i].size = num;
  partitions->partitions[i].nodeArray = nodes ;
}

void RecBisectBfLB::printPartitions(PartitionList * partitions)
{
  int i,j;
 

  CmiPrintf("\n**************************\n The Partitions are: \n");
  for (i=0; i<partitions->max; i++) {
    CmiPrintf("[%d] (%d) : \t", i, partitions->partitions[i].size);
    for (j=0; j< partitions->partitions[i].size; j++)
      CmiPrintf("%d ", partitions->partitions[i].nodeArray[j]);
    CmiPrintf("\n");
  }
}

void RecBisectBfLB::recursivePartition(int numParts, Graph *g, 
				       int nodes[], int numNodes, 
				       PartitionList *partitions) 
{
  int *p1, *p2;
  int first, second;
  int numP1, numP2;
  
  if (numParts < 2) {
    addPartition(partitions, nodes, numNodes);
  } else {
    first = numParts/2;
    second = numParts - first;
    partitionInTwo(g, nodes, numNodes, &p1, &numP1, &p2, &numP2, first,second);
    recursivePartition(first, g, p1, numP1, partitions);
    recursivePartition(second, g, p2, numP2,  partitions);
    free(nodes);
  }  
}

#endif

