/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <BaseLB.h>
#include "NullLB.h"

void CreateNullLB(void)
{
  CProxy_NullLB::ckNew();
}

static void initNullLB(void) {
#if CMK_LBDB_ON
//  LBSetDefaultCreate(CreateNullLB);
#endif
}

#if CMK_LBDB_ON
void NullLB::init(void)
{
  // if (CkMyPe() == 0) CkPrintf("[%d] NullLB created\n",CkMyPe());
  CkpvAccess(hasNullLB) = 1;
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
}

NullLB::~NullLB()
{
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->RemoveLocalBarrierReceiver(receiver);
}

void NullLB::staticAtSync(void* data)
{
  // if there is other LBs, just return
  // CmiPrintf("numLoadBalancers = %d\n", CkpvAccess(numLoadBalancers));
  if (CkpvAccess(numLoadBalancers) > 1) return;

  NullLB *me = (NullLB*)(data);
  me->AtSync();
}

void NullLB::AtSync()
{
  //Reset the database so it doesn't waste memory
  theLbdb->ClearLoads();
  
  //We don't *do* any migrations, so they're already done!
  thisProxy[CkMyPe()].migrationsDone();
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
NullLB::~NullLB() {}
void NullLB::staticAtSync(void* data) {}
void NullLB::AtSync() {}
void NullLB::migrationsDone(void) {}
#endif

#include "NullLB.def.h"
