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
#include "Refiner.h"

#include "RandRefLB.h"

CreateLBFunc_Def(RandRefLB);

static void lbinit(void) {
//        LBSetDefaultCreate(CreateRandRefLB);
  LBRegisterBalancer("RandRefLB", CreateRandRefLB, AllocateRandRefLB, "Apply random, then refine");
}

#include "RandRefLB.def.h"

RandRefLB::RandRefLB(const CkLBOptions &opt): RandCentLB(opt)
{
  lbname = (char *)"RandRefLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandRefLB created\n",CkMyPe());
}

void RandRefLB::work(CentralLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] RandRefLB strategy\n",CkMyPe());

  CkVec<MigrateInfo*> migrateInfo;
  int obj;

  RandCentLB::work(stats, count);

  // from_proc after first lb strategy
  int* from_procs = Refiner::AllocProcs(count,stats);
  for(obj=0; obj < stats->n_objs; obj++)
      from_procs[obj] = stats->to_proc[obj];

  int* to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.02);
  refiner.Refine(count,stats,from_procs,to_procs);

  for(obj=0; obj < stats->n_objs; obj++) {
      if (stats->from_proc[obj] != to_procs[obj]) {
	// CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	// 		 CkMyPe(),obj,stats->from_proc[obj],to_procs[obj]);
        stats->to_proc[obj] = to_procs[obj];
      }
  }

  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
};


/*@}*/
