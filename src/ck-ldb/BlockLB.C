
#include "BlockLB.decl.h"
#include "BlockLB.h"
#include <math.h>

extern int quietModeRequested;

CreateLBFunc_Def (BlockLB, "Allocate objects in blocks to the remaining valid PE")

/**************************************************************************
**
*/
BlockLB::BlockLB (const CkLBOptions &opt) : CBase_BlockLB (opt)
{
  lbname = "BlockLB";

  if (CkMyPe () == 0 && !quietModeRequested) {
    CkPrintf("CharmLB> BlockLB created.\n");
  }
}


/**************************************************************************
**
*/
bool BlockLB::QueryBalanceNow (int _step)
{
  return true;
}


/**************************************************************************
**
*/
void BlockLB::work (LDStats *stats)
{
  int proc;
  int obj;
  int dest;
  LDObjData *odata;


  // Make sure that there is at least one available processor.
  int validProcs=0;
  int *mapValidToAbsolute = new int[stats->nprocs()];
  for (proc = 0; proc < stats->nprocs(); proc++) {
    if (stats->procs[proc].available) {
			mapValidToAbsolute[validProcs] = proc;
			validProcs++;
    }
  }
  if (validProcs == 0) {
    CmiAbort ("BlockLB: no available processors!");
  }

	int objsPerProcessor_floor = stats->n_objs/validProcs;
  int remainder = stats->n_objs % validProcs;
  int markidx = remainder * (objsPerProcessor_floor + 1);


  int validDest;
  // Rotate each object to the next higher processor.
  for (obj = 0; obj < stats->n_objs; obj++) {
    odata = &(stats->objData[obj]);
		const CmiUInt8& objID = odata->objID();
    if (odata->migratable) {
      CkGroupID locMgrGid;
      locMgrGid.idx = odata->omHandle().id.id.idx;
      CkLocMgr *localLocMgr = (CkLocMgr *) CkLocalBranch(locMgrGid);
      CkArrayIndex arr_idx = localLocMgr->lookupIdx(objID);
      int idx = arr_idx.data()[0];
      if (idx < markidx) {
        validDest = idx / (objsPerProcessor_floor + 1);
      } else {
        validDest = ((idx - markidx) / objsPerProcessor_floor) + remainder;
      }
			dest = mapValidToAbsolute[validDest];
			stats->to_proc[obj] = dest;
			//printf("Index %d object shows up in BlockLB moved to %d\n",objID.id[0],stats->to_proc[obj]);
    }
  }

	delete [] mapValidToAbsolute;
}


#include "BlockLB.def.h"
