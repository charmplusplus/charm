#include "cksyncbarrier.h"

CkGroupID _syncBarrier;

CkpvDeclare(bool, cksyncbarrierInited);

void _cksyncbarrierInit()
{
  CkpvInitialize(bool, cksyncbarrierInited);
  CkpvAccess(cksyncbarrierInited) = false;
}

// mainchare
CkSyncBarrierInit::CkSyncBarrierInit(CkArgMsg* m)
{
  _syncBarrier = CProxy_CkSyncBarrier::ckNew();
  delete m;
}

void CkSyncBarrier::reset()
{
  if (rank0pe)
  {
    std::fill(rank_needs_flood.begin(), rank_needs_flood.end(), true);
    received_from_left = false;
    received_from_right = false;
  }
  else
    received_from_rank0 = false;
}

// Since AtSync() is global across all registered objects, the epoch should be consistent
// across PEs. The incoming client might have called AtSync() before it gets migrated in, so
// track it and check the barrier if necessary.
LDBarrierClient CkSyncBarrier::AddClient(Chare* chare, std::function<void()> fn,
                                         int epoch)
{
  if (epoch == -1)
    epoch = cur_epoch;
  else if (epoch > cur_epoch)
  {
    // If the incoming client is ahead, then record those syncs and check the barrier if
    // it can trigger. Do this asynchronously so that the caller functions for object
    // construction finish first.
    at_count += epoch - cur_epoch;
    if (at_count >= clients.size())
      thisProxy[thisIndex].CheckBarrier(false);
  }

  LBClient* new_client = new LBClient(chare, fn, epoch);
  return LDBarrierClient(clients.insert(clients.end(), new_client));
}

void CkSyncBarrier::RemoveClient(LDBarrierClient c)
{
  const auto epoch = (*c)->epoch;
  if (epoch > cur_epoch)
    at_count -= epoch - cur_epoch;
  delete *(c);
  clients.erase(c);
  if (at_count >= clients.size())
      thisProxy[thisIndex].CheckBarrier(false);
}

LDBarrierReceiver CkSyncBarrier::AddReceiver(std::function<void()> fn)
{
  LBReceiver* new_receiver = new LBReceiver(fn);

  return LDBarrierReceiver(receivers.insert(receivers.end(), new_receiver));
}

LDBarrierReceiver CkSyncBarrier::AddEndReceiver(std::function<void()> fn)
{
  LBReceiver* new_receiver = new LBReceiver(fn);
  return LDBarrierReceiver(endReceivers.insert(endReceivers.end(), new_receiver));
}

void CkSyncBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  delete *(c);
  receivers.erase(c);
}

void CkSyncBarrier::RemoveEndReceiver(LDBarrierReceiver c)
{
  delete *(c);
  endReceivers.erase(c);
}

void CkSyncBarrier::TurnOnReceiver(LDBarrierReceiver c) { (*c)->on = 1; }

void CkSyncBarrier::TurnOffReceiver(LDBarrierReceiver c) { (*c)->on = 0; }

void CkSyncBarrier::AtBarrier(LDBarrierClient h, bool flood_atsync)
{
  (*h)->epoch++;
  at_count++;

  CheckBarrier(flood_atsync);
}

void CkSyncBarrier::DecreaseBarrier(int c) { at_count -= c; }

void CkSyncBarrier::propagate_atsync()
{
  if (propagated_atsync_step < cur_epoch)
  {
    const int mype = CkMyPe();
    const int mynode = CkNodeOf(mype);
    if (!rank0pe)
    {
      if (!received_from_rank0)
      {
        // If this PE is non-rank0 and non-empty PE, then trigger AtSync barrier on rank0
        int node_rank0_pe = CkNodeFirst(mynode);
        thisProxy[node_rank0_pe].recvLbStart(cur_epoch, mynode, mype);
      }
    }
    else
    {  // Rank0 PE
      // Flood non-zero ranks on this node
      for (int i = 1; i < rank_needs_flood.size(); ++i)
      {
        if (rank_needs_flood[i])
        {
          thisProxy[mype + i].recvLbStart(cur_epoch, mynode, mype);
        }
      }
      if (!received_from_left && mynode > 0)
      {  // Flood left node
        int pe = CkNodeFirst(mynode - 1);
        thisProxy[pe].recvLbStart(cur_epoch, mynode, mype);
      }
      if (!received_from_right && mynode < CkNumNodes() - 1)
      {  // Flood right node
        int pe = CkNodeFirst(mynode + 1);
        thisProxy[pe].recvLbStart(cur_epoch, mynode, mype);
      }
    }
    propagated_atsync_step = cur_epoch;
  }
}

void CkSyncBarrier::recvLbStart(int lb_step, int sourcenode, int pe)
{
  if (lb_step != cur_epoch || startedAtSync) return;
  const int mype = CkMyPe();
  const int mynode = CkNodeOf(mype);
  if (sourcenode < mynode)
    received_from_left = true;
  else if (sourcenode > mynode)
    received_from_right = true;
  else if (rank0pe) // convert incoming pe number to local rank
    rank_needs_flood[pe - mype] = false;
  else
    received_from_rank0 = true;
  if (clients.size() == 1 &&
      clients.front()->chare->isLocMgr())  // CkLocMgr is usually a client on each PE
    CheckBarrier(true);  // Empty PE invokes barrier on self on receiving a flood msg
}

void CkSyncBarrier::CheckBarrier(bool flood_atsync)
{
  if (!on) return;

  const auto client_count = clients.size();

  if (client_count == 1 && !flood_atsync)
  {
    if (clients.front()->chare->isLocMgr()) return;
  }

  // If there are no clients, resume as soon as we're turned on
  if (client_count == 0)
  {
    cur_epoch++;
    CallReceivers();
  }

  if (at_count >= client_count)
  {
    bool at_barrier = true;

    for (auto& c : clients)
    {
      if (c->epoch <= cur_epoch)
      {
        at_barrier = false;
        break;
      }
    }

    if (at_barrier)
    {
      startedAtSync = true;
      propagate_atsync();
      at_count -= client_count;
      cur_epoch++;
      CallReceivers();
    }
  }
}

void CkSyncBarrier::CallReceivers(void)
{
  for (auto& r : receivers)
  {
    if (r->on)
    {
      r->fn();
    }
  }
}

void CkSyncBarrier::ResumeClients(void)
{
  for (auto& er : endReceivers)
  {
    if (er->on)
    {
      er->fn();
    }
  }
  for (auto& c : clients) c->fn();

  reset();
  startedAtSync = false;
}

void CkSyncBarrier::pup(PUP::er& p)
{
  IrrGroup::pup(p);
}

#include "CkSyncBarrier.def.h"
