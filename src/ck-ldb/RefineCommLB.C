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

#include "RefineCommLB.h"

CreateLBFunc_Def(RefineCommLB);

static void lbinit(void) {
  LBRegisterBalancer("RefineCommLB", 
                     CreateRefineCommLB, 
                     "Average load among processors by moving objects away from overloaded processor, communication aware");
}

#include "RefineCommLB.def.h"

RefineCommLB::RefineCommLB(const CkLBOptions &opt): RefineLB(opt)
{
  lbname = "RefineCommLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineCommLB created\n",CkMyPe());
}

CmiBool RefineCommLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

void RefineCommLB::work(CentralLB::LDStats* stats, int count)
{
  int obj;
  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, count);

  // get original object mapping
  int* from_procs = RefinerComm::AllocProcs(count, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  int* to_procs = RefinerComm::AllocProcs(count,stats);

  RefinerComm refiner(1.003);  // overload tolerance=1.05

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
  RefinerComm::FreeProcs(from_procs);
  RefinerComm::FreeProcs(to_procs);
};

#endif


/*@}*/
