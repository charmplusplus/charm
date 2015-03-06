/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "elements.h"
#include "ckheap.h"
#include "RefineCommLB.h"

CreateLBFunc_Def(RefineCommLB, "Average load among processors by moving objects away from overloaded processor, communication aware")

RefineCommLB::RefineCommLB(const CkLBOptions &opt): CBase_RefineCommLB(opt)
{
  lbname = (char *)"RefineCommLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineCommLB created\n",CkMyPe());
}

bool RefineCommLB::QueryBalanceNow(int _step)
{
  return true;
}

void RefineCommLB::work(LDStats* stats)
{
#if CMK_LBDB_ON
  int obj;
  int n_pes = stats->nprocs();

  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, n_pes);

  // get original object mapping
  int* from_procs = RefinerComm::AllocProcs(n_pes, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  int* to_procs = RefinerComm::AllocProcs(n_pes, stats);

  RefinerComm refiner(1.003);  // overload tolerance=1.05

  refiner.Refine(n_pes, stats, from_procs, to_procs);

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
#endif
}

#include "RefineCommLB.def.h"

/*@}*/
