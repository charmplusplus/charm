/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
#include "tm_tree.h"
#include "tm_mapping.h"
#include "TreeMatchLB.h"
#include "ckgraph.h"
#include <algorithm>



CreateLBFunc_Def(TreeMatchLB, "TreeMatch load balancer, like a normal one but with empty strategy")

#include "TreeMatchLB.def.h"

TreeMatchLB::TreeMatchLB(const CkLBOptions &opt): CBase_TreeMatchLB(opt)
{
  lbname = (char*)"TreeMatchLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] TreeMatchLB created\n",CkMyPe());
}

bool TreeMatchLB::QueryBalanceNow(int _step)
{
  return true;
}


double *get_comm_speed(int *depth){
  double *res;
  int i;
  
  *depth=5;

  res=(double*)malloc(sizeof(double)*(*depth));
  res[0]=1;

  for(i=1;i<*depth;i++){
    res[i]=res[i-1]*2;
  }
  return res;
}

tm_topology_t *build_abe_topology(int nb_procs){
  int arity[4]={nb_procs,2,2,2}; /*abe memory hierarchy. Should be given by HWLOC*/
  int nbring[8]={0,2,4,6,1,3,5,7}; /*abe cores numbering. Should be given by HWLOC*/
  return build_synthetic_topology(arity,5,nbring,8);
}

class ProcLoadGreater {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.totalLoad() > p2.totalLoad());
    }
};

class ObjLoadGreater {
  public:
    bool operator()(Vertex v1, Vertex v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};

void TreeMatchLB::work(BaseLB::LDStats* stats)
{
  /** ========================= 1st Do Load Balancing =======================*/

  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);       // Processor Array
  ObjGraph *ogr = new ObjGraph(stats);          // Object Graph

  /** ============================= STRATEGY ================================ */
  parr->resetTotalLoad();

  if (_lb_args.debug()>1) 
    CkPrintf("[%d] In GreedyLB strategy\n",CkMyPe());

  int vert;

  // max heap of objects
  std::sort(ogr->vertices.begin(), ogr->vertices.end(), ObjLoadGreater());
  // min heap of processors
  std::make_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());

  for(vert = 0; vert < ogr->vertices.size(); vert++) {
    // Pop the least loaded processor
    ProcInfo p = parr->procs.front();
    std::pop_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());
    parr->procs.pop_back();

    // Increment the load of the least loaded processor by the load of the
    // 'heaviest' unmapped object
    p.totalLoad() += ogr->vertices[vert].getVertexLoad();
    ogr->vertices[vert].setNewPe(p.getProcId());

    // Insert the least loaded processor with load updated back into the heap
    parr->procs.push_back(p);
    std::push_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats


  /** ====================== 2nd do Topology aware mapping ====================*/



  int nb_procs;
  double **comm_mat;
  int i;
  int *object_mapping, *permutation;

  
  /* get number of processors and teh greedy load balancing*/
  nb_procs = stats->nprocs();
  object_mapping=stats->to_proc.getVec();
  
    
  stats->makeCommHash();
  // allocate communication matrix
  comm_mat=(double**)malloc(sizeof(double*)*nb_procs);
  for(i=0;i<nb_procs;i++){
    comm_mat[i]=(double*)calloc(nb_procs,sizeof(double));
  }
  
  /* Build the communicartion matrix*/
  for(i=0;i<stats->n_comm;i++){
    LDCommData &commData = stats->commData[i];
    if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG)){
      /* object_mapping[i] is the processors of object i*/
      int from = object_mapping[stats->getHash(commData.sender)];
      int to = object_mapping[stats->getHash(commData.receiver.get_destObj())];
      if(from!=to){
	comm_mat[from][to]+=commData.bytes;
	comm_mat[to][from]+=commData.bytes;
      }
    }
  }
  
  /* build the topology of the hardware (abe machine here)*/   
  tm_topology_t *topology=build_abe_topology(nb_procs);
  display_topology(topology);
  /* compute the affinity tree */
  tree_t *comm_tree=build_tree_from_topology(topology,comm_mat,nb_procs,NULL,NULL);
  
  /* Compute the processor permutation*/
  permutation=(int*)malloc(sizeof(int)*nb_procs);
  map_topology_simple(topology,comm_tree,permutation,NULL);


  /* 
     Apply this perutation to all objects
     Side effect: object_mapping points to the stats->to_proc.getVec() 
     So, these lines change also stats->to_proc.getVec()
  */
  for(i=0;i<nb_procs;i++)
    object_mapping[i]=permutation[object_mapping[i]];

  // free communication matrix;
  for(i=0;i<nb_procs;i++){
      free(comm_mat[i]);
  }
  free(comm_mat);
  free_topology(topology);
}

/*@}*/
