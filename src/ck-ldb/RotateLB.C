/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
** November 4, 2004
**
** This is an example Charm++ load balancer called RotateLB.  It simply
** takes each object and rotates it to the next PE in the computation.
** In addition to being a simple example load balancer, this load balancer
** is useful because it enables checking of all PUP routines throughout a
** Charm++ program.
*/

#include "RotateLB.decl.h"
#include "RotateLB.h"


CreateLBFunc_Def (RotateLB, "Rotate each object to the next higher PE")

/**************************************************************************
**
*/
RotateLB::RotateLB (const CkLBOptions &opt) : CBase_RotateLB (opt)
{
  lbname = "RotateLB";

  if (CkMyPe () == 0) {
    CkPrintf ("[%d] RotateLB created\n", CkMyPe ());
  }
}


/**************************************************************************
**
*/
bool RotateLB::QueryBalanceNow (int _step)
{
  return true;
}


/**************************************************************************
**
*/
void RotateLB::work(LDStats *stats)
{
  int proc;
  int obj;
  int dest;
  LDObjData *odata;
  int n_pes = stats->nprocs();

  // Make sure that there is at least one available processor.
  for (proc = 0; proc < n_pes; proc++) {
    if (stats->procs[proc].available) {
      break;
    }
  }
  if (proc == n_pes) {
    CmiAbort ("RotateLB: no available processors!");
  }

  // Rotate each object to the next higher processor.
  for (obj = 0; obj < stats->n_objs; obj++) {
    odata = &(stats->objData[obj]);
    if (odata->migratable) {
      dest = ((stats->from_proc[obj] + 1) % n_pes);
      while ((!stats->procs[dest].available) &&
	     (dest != stats->from_proc[obj])) {
	dest = ((dest + 1) % n_pes);
      }
      if (dest != stats->from_proc[obj]) {
	stats->to_proc[obj] = dest;
   //     CkPrintf ("[%d] Object %d is migrating from PE %d to PE %d\n",CkMyPe (), obj, stats->from_proc[obj], dest);
      }
    }
  }
}


#include "RotateLB.def.h"
