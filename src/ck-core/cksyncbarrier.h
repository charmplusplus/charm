#ifndef CKSYNCBARRIER_H
#define CKSYNCBARRIER_H

#include "CkSyncBarrier.decl.h"
#include "lbdb.h"

extern CkGroupID _syncBarrier;

class CkSyncBarrierInit : public Chare
{
public:
  CkSyncBarrierInit(CkArgMsg*);
  CkSyncBarrierInit(CkMigrateMessage* m) : Chare(m) {}
};

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

CkpvExtern(bool, cksyncbarrierInited);

class CkSyncBarrier : public CBase_CkSyncBarrier
{
private:
  std::list<LBClient*> clients;
  std::list<LBReceiver*> receivers;

  std::vector<bool> rank_needs_flood;

  int cur_refcount;
  int at_count;
  bool on;
  bool rank0pe;
  int propagated_atsync_step;
  int iter_no;
  bool received_from_left;
  bool received_from_right;
  bool received_from_rank0;
  bool startedAtSync;

  void init()
  {
    CkpvAccess(cksyncbarrierInited) = true;
    cur_refcount = 1;
    iter_no = -1;
    propagated_atsync_step = 0;
    at_count = 0;
    on = false;
    startedAtSync = false;
    rank0pe = CkMyRank() == 0;
    if (rank0pe)
    {
      rank_needs_flood.resize(CkNodeSize(CkMyNode()));
    }
    reset();
  }

  void propagate_atsync();
  void reset();
  void CallReceivers(void);

public:
  CkSyncBarrier() { init(); };
  CkSyncBarrier(CkMigrateMessage* m) : CBase_CkSyncBarrier(m) { init(); }
  ~CkSyncBarrier(){};

  void pup(PUP::er& p);

  inline static CkSyncBarrier* Object()
  {
    return CkpvAccess(cksyncbarrierInited) ? (CkSyncBarrier*)CkLocalBranch(_syncBarrier)
                                           : nullptr;
  }

  void CheckBarrier(bool flood_atsync = false);
  void recvLbStart(int lb_step, int sourcenode, int pe);

  LDBarrierClient AddClient(Chare* chare, std::function<void()> fn, int refcount = -1);
  template <typename T>
  inline LDBarrierClient AddClient(T* obj, void (T::*method)(void), int refcount = -1)
  {
    return AddClient((Chare*)obj, std::bind(method, obj), refcount);
  }

  void RemoveClient(LDBarrierClient h);
  LDBarrierReceiver AddReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver AddReceiver(T* obj, void (T::*method)(void))
  {
    return AddReceiver(std::bind(method, obj));
  }

  void RemoveReceiver(LDBarrierReceiver h);
  void TurnOnReceiver(LDBarrierReceiver h);
  void TurnOffReceiver(LDBarrierReceiver h);
  void AtBarrier(LDBarrierClient _n_c, bool flood_atsync = false);
  void DecreaseBarrier(int c);
  void TurnOn()
  {
    on = true;
    CheckBarrier();
  };
  void TurnOff() { on = false; };

  void ResumeClients(void);
};

#endif /* CKSYNCBARRIER_H */
