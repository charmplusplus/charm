/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
Status:
  * support nonmigratable attrib
  * support processor avail bitvector
*/

#include "RandCentLB.h"

CreateLBFunc_Def(RandCentLB, "Assign objects to processors randomly")

RandCentLB::RandCentLB(const CkLBOptions &opt): CBase_RandCentLB(opt)
{
  lbname = "RandCentLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RandCentLB created\n",CkMyPe());
}

bool RandCentLB::QueryBalanceNow(int _step)
{
  return true;
}

inline int chooseProc(int count)
{
  return (int)(CrnDrand()*(count-1) + 0.5);
}

void RandCentLB::work(LDStats* stats)
{
  if (_lb_args.debug()) CkPrintf("Calling RandCentLB strategy\n",CkMyPe());

  int proc, n_pes = stats->nprocs();

  for (proc=0; proc<n_pes; proc++) {
    if (stats->procs[proc].available) break;
  }
  if (proc == n_pes) CmiAbort("RandCentLB> no available processor!");

  int nmigrated = 0;
  for(int obj=0; obj < stats->n_objs; obj++) {
      LDObjData &odata = stats->objData[obj];
      if (odata.migratable) {
	int dest = chooseProc(n_pes);
	while (!stats->procs[dest].available) dest = chooseProc(n_pes);
	if (dest != stats->from_proc[obj]) {
          if (_lb_args.debug() >= 2)
            CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),obj,stats->from_proc[obj],dest);
          nmigrated ++;
	  stats->to_proc[obj] = dest;
        }
      }
  }
}

#include "RandCentLB.def.h"

/*@}*/
