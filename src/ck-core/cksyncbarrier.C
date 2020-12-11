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
LDBarrierClient CkSyncBarrier::addClient(Chare* chare, std::function<void()> fn,
                                         int epoch)
{
  if (epoch == -1)
    epoch = curEpoch;
  else if (epoch > curEpoch)
  {
    // If the incoming client is ahead, then record its syncs and check the barrier if
    // eligible. Do this asynchronously so that the AddClient's caller (which is
    // constructing the object) finishes first.
    atCount += epoch - curEpoch;
    if (on && !startedAtSync && atCount >= clients.size() + 1)
      thisProxy[thisIndex].checkBarrier();
  }

  LBClient* newClient = new LBClient(chare, std::move(fn), epoch);
  return LDBarrierClient(clients.insert(clients.end(), newClient));
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

LDBarrierReceiver CkSyncBarrier::addReceiver(std::function<void()> fn)
{
  LBReceiver* newReceiver = new LBReceiver(std::move(fn));

  return LDBarrierReceiver(receivers.insert(receivers.end(), newReceiver));
}

LDBarrierReceiver CkSyncBarrier::addEndReceiver(std::function<void()> fn)
{
  LBReceiver* newReceiver = new LBReceiver(std::move(fn));
  return LDBarrierReceiver(endReceivers.insert(endReceivers.end(), newReceiver));
}

void CkSyncBarrier::removeReceiver(LDBarrierReceiver r)
{
  delete *(r);
  receivers.erase(r);
}

void CkSyncBarrier::removeEndReceiver(LDBarrierReceiver r)
{
  delete *(r);
  endReceivers.erase(r);
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
  if (kickEpoch < curEpoch || startedAtSync)
    return;

  const int myPe = CkMyPe();
  const int myNode = CkNodeOf(myPe);
  if (sourceNode < myNode)
    receivedFromLeft = true;
  else if (sourceNode > myNode)
    receivedFromRight = true;
  else if (isRank0pe) // myNode = sourceNode, so convert incoming pe number to local rank
    rankNeedsKick[sourcePe - myPe] = false;
  else
    receivedFromRank0 = true;

  if (clients.empty())
    checkBarrier();  // Empty PE invokes barrier on self on receiving a kick
}

void CkSyncBarrier::checkBarrier()
{
  if (!on) return;

  const auto clientCount = clients.size();

  if ((clientCount == 0 && curKickEpoch < curEpoch) || startedAtSync) return;

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
      startedAtSync = true;
      // Propagate kick message to trigger barrier on PEs that don't have any AtSync
      // elements on them
      propagateKick();
      atCount -= clientCount;
      curEpoch++;
      callReceivers();
    }
  }
}

void CkSyncBarrier::callReceivers()
{
  for (const auto& r : receivers)
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

  for (const auto& er : endReceivers)
  {
    if (er->on)
    {
      er->fn();
    }
  }
  for (const auto& c : clients) c->fn();
}

void CkSyncBarrier::pup(PUP::er& p)
{
  IrrGroup::pup(p);
}

#include "CkSyncBarrier.def.h"
