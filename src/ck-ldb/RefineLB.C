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

void CreateRefineLB()
{
  loadbalancer = CProxy_RefineLB::ckNew();
  //  CkPrintf("[%d] created RefineLB %d\n",CkMyPe(),loadbalancer);
}

static void lbinit(void) {
//  LBSetDefaultCreate(CreateRefineLB);        
  LBRegisterBalancer("RefineLB", CreateRefineLB, "Move objects away from overloaded processor to reach average");
}

#include "RefineLB.def.h"

RefineLB::RefineLB()
{
  lbname = "RefineLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineLB created\n",CkMyPe());
}

CmiBool RefineLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

LBMigrateMsg* RefineLB::Strategy(CentralLB::LDStats* stats, int count)
{
  int obj, pe;

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

  CkVec<MigrateInfo*> migrateInfo;

  // Save output
  for(obj=0;obj<stats->n_objs;obj++) {
      int pe = stats->from_proc[obj];
      if (to_procs[obj] != pe) {
	// CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//	 CkMyPe(),obj,pe,to_procs[obj]);
	MigrateInfo *migrateMe = new MigrateInfo;
	migrateMe->obj = stats->objData[obj].handle;
	migrateMe->from_pe = pe;
	migrateMe->to_pe = to_procs[obj];
	migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  if (lb_debug) CmiPrintf("RefineLB> migrating %d objects.\n", migrate_count);
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  // Free the refine buffers
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);

  return msg;
};

#endif


/*@}*/
