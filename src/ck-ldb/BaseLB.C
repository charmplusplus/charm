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

#include <charm++.h>
#include <BaseLB.h>

#if CMK_LBDB_ON

BaseLB::BaseLB(const CkLBOptions &opt) {
  seqno = opt.getSeqNo();
  CkpvAccess(numLoadBalancers) ++;
/*
  if (CkpvAccess(numLoadBalancers) - CkpvAccess(hasNullLB) > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
*/
  theLbdb = CProxy_LBDatabase(lbdb).ckLocalBranch();
  lbname = "Unknown";
  // register this load balancer to LBDatabase at the sequence number
  theLbdb->addLoadbalancer(this, seqno);
}

BaseLB::~BaseLB() {
  CkpvAccess(numLoadBalancers) --;
}

void BaseLB::unregister() {
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  CkpvAccess(numLoadBalancers) --;
}

#else
BaseLB::BaseLB(const CkLBOptions &) {}
BaseLB::~BaseLB() {} 

void BaseLB::unregister() {}
#endif

#include "BaseLB.def.h"

/*@}*/
