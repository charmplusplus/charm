/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb

NullLB is a place holder load balancer, it does nothing but resume from 
atSync so that application won't hang in the case where there is no other
load balancer around.
*/
/*@{*/

#include <charm++.h>
#include <BaseLB.h>
#include "NullLB.h"
#include "ck.h"

#define NULLLB_CONVERSE                     1

void CreateNullLB(void) {
  CProxy_NullLB::ckNew(-1);
}

static CkGroupID  _theNullLB;

#if NULLLB_CONVERSE
static int _migDoneHandle;		// converse handler
// converse handler 
// send converse message to avoid QD detection
static void migrationDone(envelope *env)
{
  NullLB *lb = (NullLB*)CkLocalBranch(_theNullLB);
  lb->migrationsDone();
  CkFreeSysMsg(EnvToUsr(env));
}
#endif

static void lbinit(void) {
  LBRegisterBalancer("NullLB", CreateNullLB, NULL, "should not be shown", 0);
}

static void lbprocinit(void) {
#if NULLLB_CONVERSE
  _migDoneHandle = CkRegisterHandler((CmiHandler)migrationDone);
#endif
}

#if CMK_LBDB_ON
static void staticStartLB(void* data)
{
  CmiPrintf("[%d] LB Info: StartLB called in NullLB.\n", CkMyPe());
}

void NullLB::init()
{
  // if (CkMyPe() == 0) CkPrintf("[%d] NullLB created\n",CkMyPe());
  thisProxy = CProxy_NullLB(thisgroup);
  CkpvAccess(hasNullLB) = 1;
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
			    (void*)(this));
  theLbdb->
    AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));

  _theNullLB = thisgroup;
}

NullLB::~NullLB()
{
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  theLbdb->RemoveStartLBFn((LDStartLBFn)(staticStartLB));
}

void NullLB::staticAtSync(void* data)
{
  /// if there is other LBs, just ignore return
  // CmiPrintf("numLoadBalancers = %d\n", CkpvAccess(numLoadBalancers));
  if (CkpvAccess(numLoadBalancers) > 1) return;

  NullLB *me = (NullLB*)(data);
  me->AtSync();
}

void NullLB::AtSync()
{
  // tried to reset the database so it doesn't waste memory
  // if nobody else is here, the stat collection is not even turned on
  // so I should not have to clear loads.
//  theLbdb->ClearLoads();
  
  // disable the batsyncer if no balancer exists
  // theLbdb->SetLBPeriod(1e10);

#if ! NULLLB_CONVERSE
  // prevent this charm message from being seen by QD
  // so that the QD detection works
  CpvAccess(_qd)->create(-1);
  thisProxy[CkMyPe()].migrationsDone();
#else
  // send converse message to escape the QD detection
  envelope *env = UsrToEnv(CkAllocSysMsg());
  CmiSetHandler(env, _migDoneHandle);
  CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);
#endif
}

void NullLB::migrationsDone(void)
{
#if ! NULLLB_CONVERSE
  // prevent this charm message from being seen by QD
  CpvAccess(_qd)->process(-1);
#endif
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


/*@}*/
