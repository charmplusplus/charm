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
  startedAtSync = false;

  if (rank0pe)
  {
    std::fill(rank_needs_kick.begin(), rank_needs_kick.end(), true);
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
    if (on && !startedAtSync && at_count >= clients.size() + 1)
      thisProxy[thisIndex].CheckBarrier();
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
  if (on && !startedAtSync && at_count >= clients.size())
      thisProxy[thisIndex].CheckBarrier();
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

void CkSyncBarrier::AtBarrier(LDBarrierClient h)
{
  (*h)->epoch++;
  at_count++;

  CheckBarrier();
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
      // Kick non-zero ranks on this node
      for (int i = 1; i < rank_needs_kick.size(); ++i)
      {
        if (rank_needs_kick[i])
        {
          thisProxy[mype + i].recvLbStart(cur_epoch, mynode, mype);
        }
      }
      if (!received_from_left && mynode > 0)
      {  // Kick left node
        int pe = CkNodeFirst(mynode - 1);
        thisProxy[pe].recvLbStart(cur_epoch, mynode, mype);
      }
      if (!received_from_right && mynode < CkNumNodes() - 1)
      {  // Kick right node
        int pe = CkNodeFirst(mynode + 1);
        thisProxy[pe].recvLbStart(cur_epoch, mynode, mype);
      }
    }
    propagated_atsync_step = cur_epoch;
  }
}

void CkSyncBarrier::recvLbStart(int lb_step, int sourcenode, int pe)
{
  if (lb_step > currentKick)
    currentKick = lb_step;
  if (lb_step < cur_epoch || startedAtSync)
    return;

  const int mype = CkMyPe();
  const int mynode = CkNodeOf(mype);
  if (sourcenode < mynode)
    received_from_left = true;
  else if (sourcenode > mynode)
    received_from_right = true;
  else if (rank0pe) // convert incoming pe number to local rank
    rank_needs_kick[pe - mype] = false;
  else
    received_from_rank0 = true;

  if (clients.empty())
    CheckBarrier();  // Empty PE invokes barrier on self on receiving a kick
}

void CkSyncBarrier::CheckBarrier()
{
  if (!on) return;

  const auto client_count = clients.size();

  if ((client_count == 0 && currentKick < cur_epoch) || startedAtSync) return;

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
  // The end receiver or client functions may trigger the barrier again, so make sure
  // reset() is called before them to put the barrier in a valid state to be triggered
  reset();

  for (auto& er : endReceivers)
  {
    if (er->on)
    {
      er->fn();
    }
  }
  for (auto& c : clients) c->fn();
}

void CkSyncBarrier::pup(PUP::er& p)
{
  IrrGroup::pup(p);
}

#include "CkSyncBarrier.def.h"
