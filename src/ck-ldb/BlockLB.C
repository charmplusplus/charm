
#include "charm++.h"
#include "cklists.h"

#include "BlockLB.decl.h"

#include "BlockLB.h"


CreateLBFunc_Def (BlockLB, "Allocate objects in blocks to the remaining valid PE")

/**************************************************************************
**
*/
BlockLB::BlockLB (const CkLBOptions &opt) : CentralLB (opt)
{
  lbname = "BlockLB";

  if (CkMyPe () == 0) {
    CkPrintf ("[%d] BlockLB created\n", CkMyPe ());
  }
}


/**************************************************************************
**
*/
CmiBool BlockLB::QueryBalanceNow (int _step)
{
  return CmiTrue;
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

	int objsPerProcessor = stats->n_objs/validProcs;
  // Rotate each object to the next higher processor.
  for (obj = 0; obj < stats->n_objs; obj++) {
    odata = &(stats->objData[obj]);
		const LDObjid& objID = odata->objID();
    if (odata->migratable) {
			int idx = objID.id[0];
			int validDest = idx/objsPerProcessor;
			validDest = validDest % validProcs;
			dest = mapValidToAbsolute[validDest];
			stats->to_proc[obj] = dest;
	//		printf("Index %d object shows up in BlockLB moved to %d\n",objID.id[0],stats->to_proc[obj]);
    }
  }

	delete [] mapValidToAbsolute;
}


#include "BlockLB.def.h"
