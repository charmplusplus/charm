/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <BaseLB.h>

CkpvDeclare(int, numLoadBalancers);
CkpvDeclare(int, hasNullLB);

void _loadbalancerInit()
{
//CmiPrintf("[%d] initLBFrameWork()\n", CkMyPe());
  CkpvInitialize(int, numLoadBalancers);
  CkpvInitialize(int, hasNullLB);
  CkpvAccess(numLoadBalancers) = 0;
  CkpvAccess(hasNullLB) = 0;
}

#if CMK_LBDB_ON

BaseLB::BaseLB() {
  CkpvAccess(numLoadBalancers) ++;
//CmiPrintf("[%d] BaseLB created!\n", CkMyPe());
  if (CkpvAccess(numLoadBalancers) - CkpvAccess(hasNullLB) > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
}

void BaseLB::unregister() {
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  CkpvAccess(numLoadBalancers) --;
}
#else
BaseLB::BaseLB() {}
void BaseLB::unregister() {}
#endif

#include "BaseLB.def.h"
