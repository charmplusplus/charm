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
#include "Refiner.h"

#include "RandRefLB.h"

void CreateRandRefLB()
{
  //  CkPrintf("[%d] creating RandRefLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RandRefLB::ckNew();
  //  CkPrintf("[%d] created RandRefLB %d\n",CkMyPe(),loadbalancer);
}

static void lbinit(void) {
//        LBSetDefaultCreate(CreateRandRefLB);
  LBRegisterBalancer("RandRefLB", CreateRandRefLB, "Apply random, then refine");
}

#include "RandRefLB.def.h"

RandRefLB::RandRefLB()
{
  lbname = "RandRefLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandRefLB created\n",CkMyPe());
}

void RandRefLB::work(CentralLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] RandRefLB strategy\n",CkMyPe());

  CkVec<MigrateInfo*> migrateInfo;
  int obj;

  int* from_procs = Refiner::AllocProcs(count,stats);

  for(obj=0; obj < stats->n_objs; obj++)
      from_procs[obj] = (int)(CrnDrand()*(CkNumPes()-1) + 0.5 );

  int* to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.02);
  refiner.Refine(count,stats,from_procs,to_procs);

  for(obj=0; obj < stats->n_objs; obj++) {
      LDObjData &oData = stats->objData[obj];
      if (stats->from_proc[obj] != to_procs[obj]) {
	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
			 CkMyPe(),obj,stats->from_proc[obj],to_procs[obj]);
        stats->to_proc[obj] = to_procs[obj];
      }
  }

  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
};

#endif


/*@}*/
