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

/*
Status:
  * support nonmigratable attrib
  * does not support processor avail bitvector
*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"

#include "RandCentLB.h"

CreateLBFunc_Def(RandCentLB);

static void lbinit(void) {
  LBRegisterBalancer("RandCentLB", CreateRandCentLB, "Assign objects to processors randomly");
}
#include "RandCentLB.def.h"

RandCentLB::RandCentLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "RandCentLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

CmiBool RandCentLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

void RandCentLB::work(CentralLB::LDStats* stats, int count)
{
  int nmigrated = 0;
  for(int obj=0; obj < stats->n_objs; obj++) {
      LDObjData &odata = stats->objData[obj];
      if (odata.migratable) {
	const int dest = (int)(CrnDrand()*(count-1) + 0.5);
	if (dest != stats->from_proc[obj]) {
          //CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),obj,stats->from_proc[obj],dest);
          nmigrated ++;
	  stats->to_proc[obj] = dest;
        }
      }
  }
}

#endif


/*@}*/
