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


LBMigrateMsg* GreedyRefLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkVec<MigrateInfo*> migrateInfo;
  int obj;

  work(stats, count);    // call GreedyLB first

  // Get a new buffer to refine into
  int* from_procs = Refiner::AllocProcs(count, stats);

  for(obj=0;obj < stats->n_objs; obj++)
      from_procs[obj] = stats->to_proc[obj];

  int* to_procs = Refiner::AllocProcs(count,stats);

  Refiner refiner(1.01);  // overload tolerance=1.05
  refiner.Refine(count,stats,from_procs,to_procs);

  // Save output
  for(obj=0;obj<stats->n_objs;obj++) {
      int frompe = stats->from_proc[obj];

      if (from_procs[obj] == -1) CkPrintf("From_Proc was unassigned!\n");
      if (to_procs[obj] == -1) CkPrintf("To_Proc was unassigned!\n");
      
      if (to_procs[obj] != frompe) {
  //  CkPrintf("[%d] Obj %d migrating from %d to %d\n",
  //     CkMyPe(),obj,pe,to_procs[pe][obj]);
        //CkPrintf("Refinement moved obj %d orig %d from %d to %d\n", obj,stats->from_proc[obj],from_procs[obj],to_procs[obj]);
        MigrateInfo *migrateMe = new MigrateInfo;
        migrateMe->obj = stats->objData[obj].handle;
        migrateMe->from_pe = frompe;
        migrateMe->to_pe = to_procs[obj];
        migrateInfo.insertAtEnd(migrateMe);
      }
  }

  // Free the refine buffers
  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);
  
  int migrate_count=migrateInfo.length();
  CkPrintf("GreedyRefLB migrating %d elements\n",migrate_count);
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
  
}

#endif

/*@}*/
