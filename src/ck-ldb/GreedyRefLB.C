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

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"
#include "GreedyRefLB.h"

void CreateGreedyRefLB()
{
  //  CkPrintf("[%d] creating GreedyRefLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_GreedyRefLB::ckNew();
  //  CkPrintf("[%d] created GreedyRefLB %d\n",CkMyPe(),loadbalancer);
}

static void lbinit(void) {
//        LBSetDefaultCreate(CreateGreedyRefLB);        
  LBRegisterBalancer("GreedyRefLB", CreateGreedyRefLB, "Apply greedy, then refine");
}

#include "GreedyRefLB.def.h"

GreedyRefLB::GreedyRefLB()
{
  lbname = "GreedyRefLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] GreedyRefLB created\n",CkMyPe());
}


void GreedyRefLB::work(CentralLB::LDStats* stats, int count)
{
  int obj;

  GreedyLB::work(stats, count);    // call GreedyLB first

  // Get a new buffer to refine into
  int* from_procs = Refiner::AllocProcs(count, stats);

  for(obj=0;obj < stats->n_objs; obj++)
      from_procs[obj] = stats->to_proc[obj];

  int* to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.01);  // overload tolerance=1.05
  refiner.Refine(count,stats,from_procs,to_procs);

  // Save output
  for(obj=0;obj<stats->n_objs;obj++) {
      if (to_procs[obj] == -1) CkPrintf("To_Proc was unassigned!\n");
      stats->to_proc[obj] = to_procs[obj];
  }

  // Free the refine buffers
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
}

#endif

/*@}*/
