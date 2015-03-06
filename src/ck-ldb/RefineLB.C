/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "elements.h"
#include "ckheap.h"
#include "RefineLB.h"

CreateLBFunc_Def(RefineLB, "Move objects away from overloaded processor to reach average")

RefineLB::RefineLB(const CkLBOptions &opt): CBase_RefineLB(opt)
{
  lbname = (char *)"RefineLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineLB created\n",CkMyPe());
}

void RefineLB::work(LDStats* stats)
{
  int obj;
  int n_pes = stats->nprocs();

  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, n_pes);

  // get original object mapping
  int* from_procs = Refiner::AllocProcs(n_pes, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  int* to_procs = Refiner::AllocProcs(n_pes, stats);

  Refiner refiner(1.05);  // overload tolerance=1.05

  refiner.Refine(n_pes, stats, from_procs, to_procs);

  // Save output
  for(obj=0;obj<stats->n_objs;obj++) {
      int pe = stats->from_proc[obj];
      if (to_procs[obj] != pe) {
        if (_lb_args.debug()>=2)  {
	  CkPrintf("[%d] Obj %d migrating from %d to %d\n",
		 CkMyPe(),obj,pe,to_procs[obj]);
        }
	stats->to_proc[obj] = to_procs[obj];
      }
  }

  if (_lb_args.metaLbOn()) {
    stats->is_prev_lb_refine = 1;
    stats->after_lb_avg = refiner.computeAverageLoad();
    stats->after_lb_max = refiner.computeMax();

    if (_lb_args.debug() > 0)
      CkPrintf("RefineLB> Max load %lf Avg load %lf\n", stats->after_lb_max,
          stats->after_lb_avg);
  }

  // Free the refine buffers
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
}

#include "RefineLB.def.h"

/*@}*/
