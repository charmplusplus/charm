#include <utility>

#include "cksyncbarrier.h"

CkGroupID _syncBarrier;

CkpvDeclare(bool, CkSyncBarrierInited);

void _CkSyncBarrierInit()
{
  CkpvInitialize(bool, CkSyncBarrierInited);
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
#if LBDB_ON
LDBarrierClient CkSyncBarrier::addClient(Chare* chare, std::function<void()> fn,
                                         int epoch)
{
  if (epoch == -1)
    epoch = curEpoch;
  else if (epoch > curEpoch)
  {
    // If the incoming client is ahead of us, then record those syncs
    atCount += epoch - curEpoch;
  }

  const auto client = LDBarrierClient(
      clients.insert(clients.end(), new LBClient(chare, std::move(fn), epoch)));
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
#endif
#if LBDB_ON
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
#endif

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
     //_TRACE_END_PHASE();
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

void CkSyncBarrier::resumeClients()
{
  // The end receiver or client functions may trigger the barrier again, so make sure
  // reset() is called before them to put the barrier in a valid state to be triggered
  reset();

  callReceiverList(endReceivers);

  for (const auto& c : clients) c->fn();
}

void CkSyncBarrier::pup(PUP::er& p) { IrrGroup::pup(p); }

#include "CkSyncBarrier.def.h"
