/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"
#include "Refiner.h"

#include "RandRefLB.h"
#include "RandRefLB.def.h"

void CreateRandRefLB()
{
  //  CkPrintf("[%d] creating RandRefLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RandRefLB::ckNew();
  //  CkPrintf("[%d] created RandRefLB %d\n",CkMyPe(),loadbalancer);
}

RandRefLB::RandRefLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandRefLB created\n",CkMyPe());
}

CmiBool RandRefLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

CLBMigrateMsg* RandRefLB::Strategy(CentralLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] RandRefLB strategy\n",CkMyPe());

  CkVector migrateInfo;

  int** from_procs = Refiner::AllocProcs(count,stats);
  int pe;
  int obj;
  for(pe=0; pe < count; pe++) {
    //    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
    //	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);
    for(obj=0; obj < stats[pe].n_objs; obj++)
      from_procs[pe][obj] = static_cast<int>(CrnDrand()*(CmiNumPes()-1) 
					     + 0.5 );
  }
  int** to_procs = Refiner::AllocProcs(count,stats);
  Refiner refiner(1.02);
  refiner.Refine(count,stats,from_procs,to_procs);

  for(pe=0; pe < count; pe++) {
    for(obj=0; obj < stats[pe].n_objs; obj++) {
      if (to_procs[pe][obj] != pe) {
	//	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//		 CkMyPe(),obj,pe,dest);
	MigrateInfo* migrateMe = new MigrateInfo;
	migrateMe->obj = stats[pe].objData[obj].handle;
	migrateMe->from_pe = pe;
	migrateMe->to_pe = to_procs[pe][obj];
	migrateInfo.push_back((void*)migrateMe);
      }
    }
  }

  int migrate_count=migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  Refiner::FreeProcs(from_procs);
  Refiner::FreeProcs(to_procs);

  return msg;
};

#endif
