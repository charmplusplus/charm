/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <charm++.h>
#include <BaseLB.h>

int numLoadBalancers = 0;
int hasNullLB = 0;

#if CMK_LBDB_ON
BaseLB::BaseLB() {
  numLoadBalancers ++;
  if (numLoadBalancers - hasNullLB > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
}

void BaseLB::unregister() {
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  numLoadBalancers --;
}
#endif

#include "BaseLB.def.h"
