#include <utility>

#include "cksyncbarrier.h"

CkGroupID _syncBarrier;

CkpvDeclare(bool, CkSyncBarrierInited);

void _CkSyncBarrierInit()
{
  CkpvInitialize(bool, CkSyncBarrierInited);
#if CMK_SHRINK_EXPAND
  // On survivor restart the CkSyncBarrier group object is preserved in memory
  // (init.C gates the group-table overwrites on _reuseRegistrationStateOnRestart),
  // so its constructor — and therefore init() which sets this flag to true — does
  // not re-run. Resetting the flag here would leave object() returning nullptr,
  // and the next CentralLB::ResumeClients() would segfault dereferencing it.
  extern bool _reuseRegistrationStateOnRestart;
  if (_reuseRegistrationStateOnRestart) return;
#endif
  CkpvAccess(CkSyncBarrierInited) = false;
}

// mainchare
CkSyncBarrierInit::CkSyncBarrierInit(CkArgMsg* m)
{
  _syncBarrier = CProxy_CkSyncBarrier::ckNew();
  delete m;
}

void CkSyncBarrier::reset()
{
  startedAtSync = false;

  if (isRank0pe)
  {
    std::fill(rankNeedsKick.begin(), rankNeedsKick.end(), true);
    receivedFromLeft = false;
    receivedFromRight = false;
  }
  else
    receivedFromRank0 = false;
}

// Since AtSync() is global across all registered objects, the epoch is valid across PEs.
// The incoming client might have called AtSync() before it gets migrated in, so track it
// and check the barrier if necessary.
LDBarrierClient CkSyncBarrier::addClient(Chare* chare, std::function<void()> fn,
                                         int epoch)
{
  bool late = false;
  if (epoch == -1)
    epoch = curEpoch;
  else if (epoch > curEpoch)
  {
    // If the incoming client is ahead of us, then record those syncs
    atCount += epoch - curEpoch;
  }
  else if (epoch < curEpoch)
  {
    // Late arrival: this client already completed AtSync for a round that
    // this PE has resumed past. This happens when an element migrates onto
    // a PE whose empty barrier was kick-triggered beyond the element's
    // round before the (slower) migration message arrived — e.g. an expand
    // newcomer receiving its first elements while faster survivors already
    // started the next round. The resumeClients() for its round is never
    // coming on this PE, so resume it on arrival and align it with the
    // current round; its next AtSync then counts toward this round.
    epoch = curEpoch;
    late = true;
  }

  const auto client = LDBarrierClient(
      clients.insert(clients.end(), new LBClient(chare, std::move(fn), epoch)));
  if (late)
  {
    // Resume asynchronously so the caller's construction path (migration
    // unpacking) finishes first.
    lateClients.push_back(*client);
    thisProxy[thisIndex].resumeLateClients();
  }
  // Check the barrier if it can trigger. Do this asynchronously so that the caller
  // functions for object construction finish first.
  if (on && !startedAtSync && atCount >= clients.size())
    thisProxy[thisIndex].checkBarrier();
  return client;
}

void CkSyncBarrier::removeClient(LDBarrierClient c)
{
  const auto epoch = (*c)->epoch;
  if (epoch > curEpoch)
    atCount -= epoch - curEpoch;
  delete *(c);
  clients.erase(c);
  if (on && !startedAtSync && atCount >= clients.size())
    thisProxy[thisIndex].checkBarrier();
}

LDBarrierReceiver CkSyncBarrier::addReceiverHelper(std::function<void()> fn,
                                                   std::list<LBReceiver*>& receiverList)
{
  LBReceiver* newReceiver = new LBReceiver(std::move(fn));
  return LDBarrierReceiver(receiverList.insert(receiverList.end(), newReceiver));
}

LDBarrierReceiver CkSyncBarrier::addReceiver(std::function<void()> fn)
{
  return addReceiverHelper(std::move(fn), receivers);
}

LDBarrierReceiver CkSyncBarrier::addBeginReceiver(std::function<void()> fn)
{
  return addReceiverHelper(std::move(fn), beginReceivers);
}

LDBarrierReceiver CkSyncBarrier::addEndReceiver(std::function<void()> fn)
{
  return addReceiverHelper(std::move(fn), endReceivers);
}

void CkSyncBarrier::removeReceiverHelper(LDBarrierReceiver r,
                                         std::list<LBReceiver*>& receiverList)
{
  delete *(r);
  receiverList.erase(r);
}

void CkSyncBarrier::removeReceiver(LDBarrierReceiver r)
{
  removeReceiverHelper(r, receivers);
}

void CkSyncBarrier::removeBeginReceiver(LDBarrierReceiver r)
{
  removeReceiverHelper(r, beginReceivers);
}

void CkSyncBarrier::removeEndReceiver(LDBarrierReceiver r)
{
  removeReceiverHelper(r, endReceivers);
}

void CkSyncBarrier::turnOnReceiver(LDBarrierReceiver r) { (*r)->on = true; }

void CkSyncBarrier::turnOffReceiver(LDBarrierReceiver r) { (*r)->on = false; }

void CkSyncBarrier::atBarrier(LDBarrierClient c)
{
  (*c)->epoch++;
  atCount++;

  checkBarrier();
}

// Whenever a PE triggers the barrier, send a kick through the system to tell PEs without
// any AtSync elements on them to also trigger the barrier.
// Without this, PEs devoid of AtSync elements would never trigger their receivers, which
// would cause a hang if the receiver uses group reductions (as load balancing does, for
// example).
void CkSyncBarrier::propagateKick()
{
  const int myPe = CkMyPe();
  const int myNode = CkNodeOf(myPe);
  if (!isRank0pe)
  {  // Propagate kick to rank 0 if we haven't received from it
    if (!receivedFromRank0)
    {
      const int rank0Pe = CkNodeFirst(myNode);
      thisProxy[rank0Pe].kick(curEpoch, myNode, myPe);
    }
  }
  else
  {  // Rank 0 PE
    // Propagate kick to the rest of the ranks on this node
    for (int i = 1; i < rankNeedsKick.size(); ++i)
    {
      if (rankNeedsKick[i])
      {
        thisProxy[myPe + i].kick(curEpoch, myNode, myPe);
      }
    }
    if (!receivedFromLeft && myNode > 0)
    {  // Kick left node
      const int pe = CkNodeFirst(myNode - 1);
      thisProxy[pe].kick(curEpoch, myNode, myPe);
    }
    if (!receivedFromRight && myNode < CkNumNodes() - 1)
    {  // Kick right node
      const int pe = CkNodeFirst(myNode + 1);
      thisProxy[pe].kick(curEpoch, myNode, myPe);
    }
  }
}

void CkSyncBarrier::kick(int kickEpoch, const int sourceNode, const int sourcePe)
{
  curKickEpoch = std::max(kickEpoch, curKickEpoch);

  // Ignore the kick if it's for an epoch we've already completed or we're currently
  // triggered
  if (kickEpoch <= curEpoch || startedAtSync)
    return;

  const int myPe = CkMyPe();
  const int myNode = CkNodeOf(myPe);
  if (sourceNode < myNode)
    receivedFromLeft = true;
  else if (sourceNode > myNode)
    receivedFromRight = true;
  else if (isRank0pe)  // myNode = sourceNode, so convert incoming pe number to local rank
    rankNeedsKick[sourcePe - myPe] = false;
  else
    receivedFromRank0 = true;

  if (clients.empty())
    checkBarrier();  // Empty PE invokes barrier on self on receiving a kick
}

void CkSyncBarrier::checkBarrier()
{
  if (!on)
    return;

  const auto clientCount = clients.size();

  // If there are no clients and the current kick is out of date or we're currently in the
  // barrier, then return without triggering the barrier
  if ((clientCount == 0 && curKickEpoch <= curEpoch) || startedAtSync)
    return;

  if (atCount >= clientCount)
  {
    bool atBarrier = true;

    // Ensure that all AtSync elements on this PE have completed the current epoch before
    // triggering the barrier
    for (const auto& c : clients)
    {
      if (c->epoch <= curEpoch)
      {
        atBarrier = false;
        break;
      }
    }

    if (atBarrier)
    {
      _TRACE_END_PHASE();
      startedAtSync = true;
      curEpoch++;
      // Propagate kick message to trigger barrier on PEs that don't have any AtSync
      // elements on them
      propagateKick();
      atCount -= clientCount;
      callReceiverList(beginReceivers);
      callReceiverList(receivers);
    }
  }
}

void CkSyncBarrier::callReceiverList(const std::list<LBReceiver*>& receiverList)
{
  for (const auto& r : receiverList)
  {
    if (r->on)
    {
      r->fn();
    }
  }
}

void CkSyncBarrier::resumeLateClients()
{
  while (!lateClients.empty())
  {
    LBClient* c = lateClients.back();
    lateClients.pop_back();
    c->fn();
  }
}

void CkSyncBarrier::resumeClients()
{
  // The end receiver or client functions may trigger the barrier again, so make sure
  // reset() is called before them to put the barrier in a valid state to be triggered
  reset();

  callReceiverList(endReceivers);

  for (const auto& c : clients) c->fn();
}

void CkSyncBarrier::pup(PUP::er& p)
{
  IrrGroup::pup(p);
#if CMK_SHRINK_EXPAND
  // On expand, a newcomer constructs its CkSyncBarrier fresh with curEpoch=0,
  // but the survivors have already advanced curEpoch through every pre- and
  // post-rescale AtSync round. When the newcomer's chares hit their first
  // AtSync post-integration and the newcomer's barrier triggers, the kick
  // it propagates carries kickEpoch=1 (newcomer's freshly-incremented epoch).
  // The survivors see kickEpoch <= their own curEpoch and silently discard the
  // kick (per the "I've moved past this epoch" guard in kick()), so the
  // survivors' empty-clients barrier never fires and the cluster wedges at
  // the next regular LB step. Pup curEpoch so the newcomer adopts the
  // cluster's epoch from the rescale-time broadcast.
  p | curEpoch;
#endif
}

#include "CkSyncBarrier.def.h"
