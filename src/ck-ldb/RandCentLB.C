#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"

#include "RandCentLB.h"
#include "RandCentLB.def.h"

void CreateRandCentLB()
{
  //  CkPrintf("[%d] creating RandCentLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RandCentLB::ckNew();
  //  CkPrintf("[%d] created RandCentLB %d\n",CkMyPe(),loadbalancer);
}

RandCentLB::RandCentLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

CmiBool RandCentLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

CLBMigrateMsg* RandCentLB::Strategy(CentralLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] RandCentLB strategy\n",CkMyPe());

  CkVector migrateInfo;

  for(int pe=0; pe < count; pe++) {
    //    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
    //	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);
    for(int obj=0; obj < stats[pe].n_objs; obj++) {
      const int dest = static_cast<int>(drand48()*(CmiNumPes()-1) + 0.5);
      if (dest != pe) {
	//	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//		 CkMyPe(),obj,pe,dest);
	MigrateInfo* migrateMe = new MigrateInfo;
	migrateMe->obj = stats[pe].objData[obj].handle;
	migrateMe->from_pe = pe;
	migrateMe->to_pe = dest;
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

  return msg;
};

#endif
