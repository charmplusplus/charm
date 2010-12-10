/** \file GreedyLB.C
 *
 *  Written by Gengbin Zheng
 *  Updated by Abhinav Bhatele, 2010-12-09 to use ckgraph
 *
 *  Status:
 *    -- Does not support pe_speed's currently
 *    -- Does not support nonmigratable attribute
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "GreedyLB.h"
#include "ckgraph.h"
#include <algorithm>

CreateLBFunc_Def(GreedyLB, "always assign the heaviest obj onto lightest loaded processor.")

GreedyLB::GreedyLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "GreedyLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] GreedyLB created\n",CkMyPe());
}

CmiBool GreedyLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

class ProcLoadGreater {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

class ObjLoadGreater {
  public:
    bool operator()(Vertex v1, Vertex v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};

void GreedyLB::work(LDStats* stats)
{
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
    p.setTotalLoad(p.getTotalLoad() + ogr->vertices[vert].getVertexLoad());
    ogr->vertices[vert].setNewPe(p.getProcId());

    // Insert the least loaded processor with load updated back into the heap
    parr->procs.push_back(p);
    std::push_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats
}

#include "GreedyLB.def.h"

/*@}*/

