#include <charm++.h>

#if CMK_LBDB_ON

#if CMK_STL_USE_DOT_H
#include <deque.h>
#include <queue.h>
#else
#include <deque>
#include <queue>
#endif

#include "RandCentLB.h"
#include "RandCentLB.def.h"

#if CMK_STL_USE_DOT_H
template class deque<CentralLB::MigrateInfo>;
#else
template class std::deque<CentralLB::MigrateInfo>;
#endif

void CreateRandCentLB()
{
  CkPrintf("[%d] creating RandCentLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RandCentLB::ckNew();
  CkPrintf("[%d] created RandCentLB %d\n",CkMyPe(),loadbalancer);
}

RandCentLB::RandCentLB()
{
  CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

Bool RandCentLB::QueryBalanceNow(int step)
{
  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),step);
  return True;
}

CLBMigrateMsg* RandCentLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkPrintf("[%d] RandCentLB strategy\n",CkMyPe());

#if CMK_STL_USE_DOT_H
  queue<MigrateInfo> migrateInfo;
#else
  std::queue<MigrateInfo> migrateInfo;
#endif

  for(int pe=0; pe < count; pe++) {
    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);
    for(int obj=0; obj < stats[pe].n_objs; obj++) {
      const int dest = (int)((static_cast<double>(random()) * count) 
			     / RAND_MAX);
      if (dest != pe) {
	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
		 CkMyPe(),obj,pe,dest);
	MigrateInfo migrateMe;
	migrateMe.obj = stats[pe].objData[obj].handle;
	migrateMe.from_pe = pe;
	migrateMe.to_pe = dest;
	migrateInfo.push(migrateMe);
      }
    }
  }

  int migrate_count=migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    msg->moves[i] = migrateInfo.front();
    migrateInfo.pop();
  }

  return msg;
};

#endif
