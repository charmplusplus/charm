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
  int epoch;

  LBClient(Chare* chare, std::function<void()> fn, int epoch)
      : chare(chare), fn(fn), epoch(epoch)
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
  std::list<LBReceiver*> beginReceivers;
  std::list<LBReceiver*> endReceivers;

  std::vector<bool> rank_needs_kick;

  int cur_epoch;
  int at_count;
  bool on;
  bool rank0pe;
  int propagated_atsync_step;
  int currentKick;
  bool received_from_left;
  bool received_from_right;
  bool received_from_rank0;
  bool startedAtSync;

  void init()
  {
    CkpvAccess(cksyncbarrierInited) = true;
    cur_epoch = 1;
    currentKick = 0;
    propagated_atsync_step = 0;
    at_count = 0;
    on = false;
    rank0pe = CkMyRank() == 0;
    if (rank0pe)
    {
      rank_needs_kick.resize(CkNodeSize(CkMyNode()));
    }
    reset();
  }

  void propagate_atsync();
  void reset();
  static void CallReceiverList(const std::list<LBReceiver*>& receiverList);

  static LDBarrierReceiver AddReceiverHelper(std::function<void()> fn,
                                      std::list<LBReceiver*>& receiverList);
  void RemoveReceiverHelper(LDBarrierReceiver r, std::list<LBReceiver*>& receiverList);

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

  void CheckBarrier();
  void recvLbStart(int lb_step, int sourcenode, int pe);

  LDBarrierClient AddClient(Chare* chare, std::function<void()> fn, int epoch = -1);
  template <typename T>
  inline LDBarrierClient AddClient(T* obj, void (T::*method)(void), int epoch = -1)
  {
    return AddClient((Chare*)obj, std::bind(method, obj), epoch);
  }

  void RemoveClient(LDBarrierClient h);

  // A receiver is a callback function that is called when all of the clients on this PE
  // reach this barrier
  LDBarrierReceiver AddReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver AddReceiver(T* obj, void (T::*method)(void))
  {
    return AddReceiver(std::bind(method, obj));
  }

  // A begin receiver is a callback function that is called after all of the clients on
  // this PE reach this barrier and before calling the actual receivers, useful for
  // setting up for the execution of those receivers. Will only be called when a receiver
  // exists.
  LDBarrierReceiver AddBeginReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver AddBeginReceiver(T* obj, void (T::*method)(void))
  {
    return AddBeginReceiver(std::bind(method, obj));
  }

  // An end receiver is a callback function that is called when the receivers on this PE
  // have finished executing, right before the clients are resumed, useful for cleaning up
  // or resetting state. Will only be called when a receiver exists.
  LDBarrierReceiver AddEndReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver AddEndReceiver(T* obj, void (T::*method)(void))
  {
    return AddEndReceiver(std::bind(method, obj));
  }

  void RemoveReceiver(LDBarrierReceiver h);
  void RemoveBeginReceiver(LDBarrierReceiver h);
  void RemoveEndReceiver(LDBarrierReceiver h);
  void TurnOnReceiver(LDBarrierReceiver h);
  void TurnOffReceiver(LDBarrierReceiver h);
  void AtBarrier(LDBarrierClient _n_c);
  void DecreaseBarrier(int c);
  void TurnOn()
  {
    on = true;
    CheckBarrier();
  };
  void TurnOff() { on = false; };

  void ResumeClients(void);

  bool hasReceivers() { return !receivers.empty(); };
};

#endif /* CKSYNCBARRIER_H */
