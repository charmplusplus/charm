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

#include "RefineLB.h"

CreateLBFunc_Def(RefineLB);

static void lbinit(void) {
  LBRegisterBalancer("RefineLB", CreateRefineLB, "Move objects away from overloaded processor to reach average");
}

#include "RefineLB.def.h"

RefineLB::RefineLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "RefineLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineLB created\n",CkMyPe());
}

CmiBool RefineLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

void RefineLB::work(CentralLB::LDStats* stats, int count)
{
  int obj;
  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, count);

  // get original object mapping
  int* from_procs = Refiner::AllocProcs(count, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  int* to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.003);  // overload tolerance=1.05

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
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
};

#endif


/*@}*/
