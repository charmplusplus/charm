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

class LBClient
{
public:
  Chare* chare;
  std::function<void()> fn;
  int refcount;

  LBClient(Chare* chare, std::function<void()> fn, int refcount)
      : chare(chare), fn(fn), refcount(refcount)
  {
  }
};

class LBReceiver
{
public:
  std::function<void()> fn;
  int on;

  LBReceiver(std::function<void()> fn, int on = 1) : fn(fn), on(on) {}
};

void CkSyncBarrier::reset()
{
  if (rank0pe)
  {
    int peFirst = CkNodeFirst(CkMyNode());
    local_pes_to_notify.clear();
    for (int pe = peFirst; pe < peFirst + CkNodeSize(CkMyNode()); pe++)
      local_pes_to_notify.insert(local_pes_to_notify.end(), pe);
    received_from_left = false;
    received_from_right = false;
  }
  else
    received_from_rank0 = false;
}

LDBarrierClient CkSyncBarrier::AddClient(Chare* chare, std::function<void()> fn)
{
  LBClient* new_client = new LBClient(chare, fn, cur_refcount);

  return LDBarrierClient(clients.insert(clients.end(), new_client));
}

void CkSyncBarrier::RemoveClient(LDBarrierClient c)
{
  delete *(c);
  clients.erase(c);
}

LDBarrierReceiver CkSyncBarrier::AddReceiver(std::function<void()> fn)
{
  LBReceiver* new_receiver = new LBReceiver(fn);

  return LDBarrierReceiver(receivers.insert(receivers.end(), new_receiver));
}

void CkSyncBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  delete *(c);
  receivers.erase(c);
}

void CkSyncBarrier::TurnOnReceiver(LDBarrierReceiver c) { (*c)->on = 1; }

void CkSyncBarrier::TurnOffReceiver(LDBarrierReceiver c) { (*c)->on = 0; }

void CkSyncBarrier::AtBarrier(LDBarrierClient h, bool flood_atsync)
{
  (*h)->refcount++;
  at_count++;

  CheckBarrier(flood_atsync);
}

void CkSyncBarrier::DecreaseBarrier(int c) { at_count -= c; }

void CkSyncBarrier::propagate_atsync()
{
  if (propagated_atsync_step < cur_refcount)
  {
    int mype = CkMyPe();
    int mynode = CkNodeOf(mype);
    if (!rank0pe)
    {
      if (!received_from_rank0)
      {
        // If this PE is non-rank0 and non-empty PE, then trigger AtSync barrier on rank0
        int node_rank0_pe = CkNodeFirst(mynode);
        thisProxy[node_rank0_pe].recvLbStart(cur_refcount, mynode, mype);
      }
    }
    else
    {  // Rank0 PE
      int peFirst = CkNodeFirst(CkMyNode());
      // Flood non-zero ranks on this node
      for (std::list<int>::iterator it = local_pes_to_notify.begin();
           it != local_pes_to_notify.end(); ++it)
        thisProxy[*it].recvLbStart(cur_refcount, mynode, mype);
      if (!received_from_left && mynode > 0)
      {  // Flood left node
        int pe = CkNodeFirst(mynode - 1);
        thisProxy[pe].recvLbStart(cur_refcount, mynode, mype);
      }
      if (!received_from_right && mynode < CkNumNodes() - 1)
      {  // Flood right node
        int pe = CkNodeFirst(mynode + 1);
        thisProxy[pe].recvLbStart(cur_refcount, mynode, mype);
      }
    }
    propagated_atsync_step = cur_refcount;
  }
}

void CkSyncBarrier::recvLbStart(int lb_step, int sourcenode, int pe)
{
  if (lb_step != cur_refcount || startedAtSync) return;
  int mype = CkMyPe();
  int mynode = CkNodeOf(mype);
  if (sourcenode < mynode)
    received_from_left = true;
  else if (sourcenode > mynode)
    received_from_right = true;
  else if (rank0pe)
    local_pes_to_notify.remove(pe);
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
    cur_refcount++;
    CallReceivers();
  }

  if (at_count >= client_count)
  {
    bool at_barrier = true;

    for (auto& c : clients)
    {
      if (c->refcount < cur_refcount)
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
      cur_refcount++;
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
  for (auto& c : clients) c->fn();

  reset();
  startedAtSync = false;
}

#include "CkSyncBarrier.def.h"
