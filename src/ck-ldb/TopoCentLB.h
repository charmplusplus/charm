#ifndef _TOPOCENTLB_H_
#define _TOPOCENTLB_H_

#include "CentralLB.h"
#include "topology.h"


extern "C" void METIS_PartGraphRecursive (int*, int*, int*, int*, int*, int*,
					  int*, int*, int*, int*, int*);

extern "C" void METIS_PartGraphKway (int*, int*, int*, int*, int*, int*,
				     int*, int*, int*, int*, int*);

extern "C" void METIS_PartGraphVKway (int*, int*, int*, int*, int*, int*,
				      int*, int*, int*, int*, int*);

extern "C" void METIS_WPartGraphRecursive (int*, int*, int*, int*,
					   int*, int*, int*, int*,
					   float*, int*, int*, int*);

extern "C" void METIS_WPartGraphKway (int*, int*, int*, int*,
				      int*, int*, int*, int*,
				      float*, int*, int*, int*);

extern "C" void METIS_mCPartGraphRecursive (int*, int*, int*, int*,
					    int*, int*, int*, int*,
					    int*, int*, int*, int*);

extern "C" void METIS_mCPartGraphKway (int*, int*, int*, int*, int*,
				       int*, int*, int*, int*, int*,
				       int*, int*, int*);



void CreateTopoCentLB ();

class TopoCentLB : public CBase_TopoCentLB
{
  public:
    TopoCentLB (const CkLBOptions &opt);
    TopoCentLB (CkMigrateMessage *m) : CBase_TopoCentLB (m) { };
    ~TopoCentLB();
    
    void work (LDStats *stats);

    void pup (PUP::er &p) { }

    struct HeapNode {
      double key;
      int node;
    };

    class PartGraph {
    public:
      typedef struct{
        int degree;
        int *obj_list;
        int num_objs;
        double comm;  //Amount of communication done by this partition -- num of bytes
      } Node;
      
      typedef double Edge;

      PartGraph(int num_parts,int init_num_objs){
        int i;
        n_nodes = num_parts;
        nodes = new Node[num_parts];
        for(i=0;i<num_parts;i++)
        {
          nodes[i].obj_list = new int[init_num_objs];
          nodes[i].num_objs=0;
          nodes[i].degree=0;
          nodes[i].comm=0;
        }
        
        n_edges = num_parts*num_parts;
        edges = new Edge*[num_parts];
        for(i=0;i<num_parts;i++)
        {
          edges[i] = new Edge[num_parts];
          for(int j=0;j<num_parts;j++)
            edges[i][j] = 0;
        }
      }

      PartGraph(PartGraph *pg,int init_num_objs){
        int i;
        n_nodes = pg->n_nodes;
        n_edges = pg->n_edges;
        nodes = new Node[n_nodes];
        for(i=0;i<n_nodes;i++){
          nodes[i].obj_list=new int[init_num_objs];
          nodes[i].num_objs = pg->nodes[i].num_objs;
          nodes[i].degree = pg->nodes[i].degree;
          nodes[i].comm = pg->nodes[i].comm;
          for(int j=0;j<pg->nodes[i].num_objs;j++)
            nodes[i].obj_list[j] = pg->nodes[i].obj_list[j];
        }
        
        edges = new Edge*[n_nodes];
        for(i=0;i<n_nodes;i++){
          edges[i] = new Edge[n_nodes];
          for(int j=0;j<n_nodes;j++)
            edges[i][j] = pg->edges[i][j];
        }
      }

      ~PartGraph(){
        for(int i=0;i<n_nodes;i++)
          delete[] nodes[i].obj_list;
        delete[] nodes;
        
        for(int i=0;i<n_nodes;i++)
          delete[] edges[i];
        delete[] edges;
      }
    //private:
      Node *nodes;
      Edge **edges;
      int n_nodes;
      int n_edges;
    };
    
    PartGraph *partgraph;
    LBTopology      *topo;
    double **hopCount;
    int *heapMapping;
    //int **topoGraph;
    //int *topoDegree;
    
    void calculateMST(PartGraph *partgraph,LBTopology *topo,int *proc_mapping,int max_comm_part);
    void increaseKey(HeapNode *heap,int i,double wt);
    HeapNode extractMax(HeapNode *heap,int *heapSize);
    void BuildHeap(HeapNode *heap,int heapSize);
    void Heapify(HeapNode *heap, int node, int heapSize);
    int findMaxObjs(int *map,int totalobjs,int count);
    void computePartitions(CentralLB::LDStats *stats,int count,int *newmap);
    //static int compare(const void *p,const void *q);
  private:
    
    bool QueryBalanceNow (int step);
};

#endif /* _TOPOCENTLB_H_ */
