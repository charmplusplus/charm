#ifndef CKSYNCBARRIER_H
#define CKSYNCBARRIER_H

#include <utility>

#include "CkSyncBarrier.decl.h"
#include "lbdb.h"

extern CkGroupID _syncBarrier;

class CkSyncBarrierInit : public Chare
{
public:
  CkSyncBarrierInit(CkArgMsg* m);
  CkSyncBarrierInit(CkMigrateMessage* m) : Chare(m) {}
};

class LBClient
{
public:
  Chare* chare;
  std::function<void()> fn;
  int epoch;

  LBClient(Chare* chare, std::function<void()> fn, int epoch)
      : chare(chare), fn(std::move(fn)), epoch(epoch)
  {
  }
};

class LBReceiver
{
public:
  std::function<void()> fn;
  bool on;

  LBReceiver(std::function<void()> fn, bool on = true) : fn(std::move(fn)), on(on) {}
};

CkpvExtern(bool, CkSyncBarrierInited);

class CkSyncBarrier : public CBase_CkSyncBarrier
{
private:
  std::list<LBClient*> clients;
  std::list<LBReceiver*> receivers;
  std::list<LBReceiver*> beginReceivers;
  std::list<LBReceiver*> endReceivers;

  std::vector<bool> rankNeedsKick;

  int atCount = 0;
  int curEpoch = 0;
  int curKickEpoch = 0;
  bool on = false;
  bool isRank0pe = CkMyRank() == 0;
  bool receivedFromLeft = false;
  bool receivedFromRight = false;
  bool receivedFromRank0 = false;
  bool startedAtSync = false;

  void init()
  {
    CkpvAccess(CkSyncBarrierInited) = true;
    if (isRank0pe)
    {
      rankNeedsKick.resize(CkNodeSize(CkMyNode()), true);
    }
  }

  void propagateKick();
  void reset();
  static void callReceiverList(const std::list<LBReceiver*>& receiverList);

  static LDBarrierReceiver addReceiverHelper(std::function<void()> fn,
                                             std::list<LBReceiver*>& receiverList);
  static void removeReceiverHelper(LDBarrierReceiver r,
                                   std::list<LBReceiver*>& receiverList);

public:
  CkSyncBarrier() { init(); };
  CkSyncBarrier(CkMigrateMessage* m) : CBase_CkSyncBarrier(m) { init(); }
  ~CkSyncBarrier() override = default;

  CkSyncBarrier(const CkSyncBarrier&) = delete;
  CkSyncBarrier& operator=(const CkSyncBarrier&) = delete;
  CkSyncBarrier(CkSyncBarrier&&) = delete;
  CkSyncBarrier& operator=(CkSyncBarrier&&) = delete;

  void pup(PUP::er& p) override;

  inline static CkSyncBarrier* object()
  {
    return CkpvAccess(CkSyncBarrierInited)
               ? static_cast<CkSyncBarrier*>(CkLocalBranch(_syncBarrier))
               : nullptr;
  }

  void checkBarrier();
  void kick(int kickEpoch, int sourceNode, int sourcePe);

  LDBarrierClient addClient(Chare* chare, std::function<void()> fn, int epoch = -1);
  template <typename T>
  inline LDBarrierClient addClient(T* obj, void (T::*method)(), int epoch = -1)
  {
    return addClient((Chare*)obj, std::bind(method, obj), epoch);
  }

  void removeClient(LDBarrierClient c);

  // A receiver is a callback function that is called when all of the clients on this PE
  // reach this barrier
  LDBarrierReceiver addReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver addReceiver(T* obj, void (T::*method)())
  {
    return addReceiver(std::bind(method, obj));
  }

  // A begin receiver is a callback function that is called after all of the clients on
  // this PE reach this barrier and before calling the actual receivers, useful for
  // setting up for the execution of those receivers. Will only be called when a receiver
  // exists.
  LDBarrierReceiver addBeginReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver addBeginReceiver(T* obj, void (T::*method)(void))
  {
    return addBeginReceiver(std::bind(method, obj));
  }

  // An end receiver is a callback function that is called when the receivers on this PE
  // have finished executing, right before the clients are resumed, useful for cleaning up
  // or resetting state. Will only be called when a receiver exists.
  LDBarrierReceiver addEndReceiver(std::function<void()> fn);
  template <typename T>
  inline LDBarrierReceiver addEndReceiver(T* obj, void (T::*method)())
  {
    return addEndReceiver(std::bind(method, obj));
  }

  void removeReceiver(LDBarrierReceiver r);
  void removeBeginReceiver(LDBarrierReceiver r);
  void removeEndReceiver(LDBarrierReceiver r);
  static void turnOnReceiver(LDBarrierReceiver r);
  static void turnOffReceiver(LDBarrierReceiver r);
  void atBarrier(LDBarrierClient c);
  void turnOn()
  {
    on = true;
    checkBarrier();
  };
  void turnOff() { on = false; };

  void resumeClients();

  bool hasReceivers() { return !receivers.empty(); };
};

#endif /* CKSYNCBARRIER_H */
