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

#include "cklists.h"

#include "RefineKLB.h"

CreateLBFunc_Def(RefineKLB, "Move objects away from overloaded processor to reach average");

RefineKLB::RefineKLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = (char *)"RefineKLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineKLB created\n",CkMyPe());
}

void RefineKLB::work(BaseLB::LDStats* stats, int count)
{
  int obj;
  //  CkPrintf("[%d] RefineKLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, count);

  // get original object mapping
  //int* from_procs = Refiner::AllocProcs(count, stats);
  int* from_procs = RefinerApprox::AllocProcs(count, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  //int* to_procs = Refiner::AllocProcs(count,stats);
  int* to_procs = RefinerApprox::AllocProcs(count,stats);

  //Refiner refiner(1.003);  // overload tolerance=1.05
  RefinerApprox refiner(1.003);  // overload tolerance=1.05

  refiner.Refine(count,stats,from_procs,to_procs);

  // Save output
  for(obj=0;obj<stats->n_objs;obj++) {
      int pe = stats->from_proc[obj];
      if (to_procs[obj] != pe) {
	// CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//	 CkMyPe(),obj,pe,to_procs[obj]);
	stats->to_proc[obj] = to_procs[obj];
      }
  }

  // Free the refine buffers
  //Refiner::FreeProcs(from_procs);
  //Refiner::FreeProcs(to_procs);
  RefinerApprox::FreeProcs(from_procs);
  RefinerApprox::FreeProcs(to_procs);
};

#include "RefineKLB.def.h"

/*@}*/
