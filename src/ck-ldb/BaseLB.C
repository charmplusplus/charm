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

void BaseLB::initLB(const CkLBOptions &opt) {
  seqno = opt.getSeqNo();
  CkpvAccess(numLoadBalancers) ++;
/*
  if (CkpvAccess(numLoadBalancers) - CkpvAccess(hasNullLB) > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
*/
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
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

void BaseLB::pup(PUP::er &p) { 
  IrrGroup::pup(p); 
  p|seqno;
  if (p.isUnpacking())
  {
    if (CkMyPe()==0) {
      if (seqno!=-1) {
        int newseq = LBDatabaseObj()->getLoadbalancerTicket();
        CmiAssert(newseq == seqno);
      }
    }
    initLB(seqno);
  }
}

void BaseLB::flushStates() {
  theLbdb->ClearLoads();
}

#else
BaseLB::~BaseLB() {} 
void BaseLB::initLB(const CkLBOptions &) {}
void BaseLB::unregister() {}
void BaseLB::pup(PUP::er &p) {}
void BaseLB::flushStates() {}
#endif

#include "BaseLB.def.h"

/*@}*/
