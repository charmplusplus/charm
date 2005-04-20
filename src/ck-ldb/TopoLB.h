#ifndef _TOPOLB_H_
#define _TOPOLB_H_

#include "CentralLB.h"
#include "topology.h"

#ifndef INF
#define INF 999999
#endif

void CreateTopoLB ();

class TopoLB : public CentralLB
{
  public:
    TopoLB (const CkLBOptions &opt);
    TopoLB (CkMigrateMessage *m) : CentralLB (m) { };
  
    void work (CentralLB::LDStats *stats, int count);
    void pup (PUP::er &p) { CentralLB::pup(p); }
    	
    LBTopology			*topo;
  
  private:

    double **dist;
    double **comm;
    double *commUA;
    double **hopBytes;
    bool *pfree;
    bool *cfree;
    int *assign;
    
    void computePartitions(CentralLB::LDStats *stats,int count,int *newmap);
    void allocateDataStructures(int num_procs);
    void freeDataStructures(int num_procs);
    void initDataStructures(CentralLB::LDStats *stats,int count,int *newmap);
    void printDataStructures(int num_procs, int num_objs, int *newmap);
    
    CmiBool QueryBalanceNow (int step);
}; 












/**************************************************
       A class representing partition of compute graph 
 **************************************************/
   
/*
class TopoLB::PartGraph 
{  
public:
  typedef struct
  {
    int degree;
    int *obj_list;
    int num_objs;
    double comm;	//Amount of communication done by this partition -- num of bytes
  } Node;
	
  typedef double Edge;

  PartGraph(int num_parts,int init_num_objs)
  {
    n_nodes = num_parts;
    nodes = new Node[num_parts];
    for(int i=0;i<num_parts;i++)
    {
      nodes[i].obj_list = new int[init_num_objs];
      nodes[i].num_objs=0;
      nodes[i].degree=0;
      nodes[i].comm=0;
    }

    n_edges = num_parts*num_parts;
    edges = new Edge*[num_parts];
    for(int i=0;i<num_parts;i++)
    {
      edges[i] = new Edge[num_parts];
      for(int j=0;j<num_parts;j++)
        edges[i][j] = 0;
    }
  }

PartGraph(PartGraph *pg,int init_num_objs)
{
  n_nodes = pg->n_nodes;
  n_edges = pg->n_edges;
  nodes = new Node[n_nodes];

  for(int i=0;i<n_nodes;i++)
  {
    nodes[i].obj_list=new int[init_num_objs];
    nodes[i].num_objs = pg->nodes[i].num_objs;
    nodes[i].degree = pg->nodes[i].degree;
    nodes[i].comm = pg->nodes[i].comm;
    for(int j=0;j<pg->nodes[i].num_objs;j++)
      nodes[i].obj_list[j] = pg->nodes[i].obj_list[j];
  }

  edges = new Edge*[n_nodes];
  for(int i=0;i<n_nodes;i++)
  {
    edges[i] = new Edge[n_nodes];
    for(int j=0;j<n_nodes;j++)
      edges[i][j] = pg->edges[i][j];
  }
}

~PartGraph()
{
  for(int i=0;i<n_nodes;i++)
    delete[] nodes[i].obj_list;
  delete[] nodes;
  delete[] edges;
}
//private:
  Node *nodes;
  Edge **edges;
  int n_nodes;
  int n_edges;
};
*/

#endif /* _TOPOLB_H_ */
