/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <LBDatabase.h>
#include "NullLB.h"

void CreateNullLB(void)
{
  CProxy_NullLB::ckNew();
}

static void initNullLB(void) {
#if CMK_LBDB_ON
  LBSetDefaultCreate(CreateNullLB);
#endif
}

#if CMK_LBDB_ON
void NullLB::init(void)
{
  thisproxy=thisgroup;
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
}

void NullLB::staticAtSync(void* data)
{
  NullLB *me = (NullLB*)(data);
  me->AtSync();
}

void NullLB::AtSync()
{
  //Reset the database so it doesn't waste memory
  theLbdb->ClearLoads();
  
  //We don't *do* any migrations, so they're already done!
  thisproxy[CkMyPe()].migrationsDone();
}
void NullLB::migrationsDone(void)
{
  theLbdb->ResumeClients();
}
#else
/*No load balancer-- still need definitions to prevent linker errors.
I sure wish we had #ifdefs in the .ci file-- then we could avoid all this.
*/
void NullLB::init(void) {}
void NullLB::staticAtSync(void* data) {}
void NullLB::AtSync() {}
void NullLB::migrationsDone(void) {}
#endif

#include "NullLB.def.h"
