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

#include "RandCentLB.h"

void CreateRandCentLB()
{
  //  CkPrintf("[%d] creating RandCentLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RandCentLB::ckNew();
  //  CkPrintf("[%d] created RandCentLB %d\n",CkMyPe(),loadbalancer);
}

static void lbinit(void) {
//	LBSetDefaultCreate(CreateRandCentLB);
  LBRegisterBalancer("RandCentLB", CreateRandCentLB, "Assign objects to processors randomly");
}
#include "RandCentLB.def.h"

RandCentLB::RandCentLB()
{
  lbname = "RandCentLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

CmiBool RandCentLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

void RandCentLB::work(CentralLB::LDStats* stats, int count)
{
  for(int obj=0; obj < stats->n_objs; obj++) {
      LDObjData &odata = stats->objData[obj];
      if (odata.migratable) {
	const int dest = (int)(CrnDrand()*(count-1) + 0.5);
	if (dest != stats->from_proc[obj]) {
	  // CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),obj,stats->from_proc[obj],dest);
	  stats->to_proc[obj] = dest;
        }
      }
  }
}

#endif


/*@}*/
