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
  // The application iteration this client last arrived at, from its
  // checkRescale() call, or -1 when it never made one. The barrier itself
  // counts epochs, not iterations, so without this a rescale commit cannot
  // tell an element parked at the tentative ceiling (safe to put back to
  // work) from one that arrived at its own regular load balancing boundary
  // (must stay put). See CkSyncBarrier::rewindArrivedClients.
  int arrivedAtIter = -1;

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
  // Clients that arrived (via migration) with an epoch this PE has already
  // resumed past; they missed their round's resumeClients() and are resumed
  // asynchronously on arrival. See addClient().
  std::vector<LBClient*> lateClients;
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
#if CMK_SHRINK_EXPAND
  // A round fired and its clients have not been let go yet.
  //
  // Ordinarily startedAtSync says this, but it says several things at once and
  // the rescale reset has to clear it (so the next AtSync round can fire)
  // while this particular fact must survive: hold-boundary mode exists to
  // carry a fired-but-unresumed round across the cut, and the clients are
  // released on the far side. Once the barrier fires, every client's epoch
  // equals curEpoch and atCount is back to zero, so nothing else in the
  // barrier's state distinguishes "waiting to be resumed" from "idle".
  bool clientsAwaitingResume = false;
#endif

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
  // Resume clients only when a full local barrier is actually pending
  // (startedAtSync: every client arrived and the round began, and reset()
  // has not run). resumeClients() itself is not idempotent -- firing it on
  // running clients calls spurious ResumeFromSync on every element. Returns
  // whether a resume happened. Used by the post-rescale restore, where
  // clients may be running (early release / barrier-less), all held (the
  // hold-boundary path), or held by a racing LB round the cut beheaded.
  bool resumeClientsIfHeld();
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
  void atBarrier(LDBarrierClient c, int iter = -1);
  void turnOn()
  {
    on = true;
    checkBarrier();
  };
  void turnOff() { on = false; };
  bool isOn() const { return on; }

#if CMK_SHRINK_EXPAND
  // Survivor restart: the iter-N LB step that triggered the rescale fired the
  // begin-receivers (RegisteringObjects → turnOff) and set startedAtSync=true
  // inside checkBarrier, but the matching ResumeClients → reset() never ran
  // (the rescale path forks off at CheckForRealloc → StartCleanup → longjmp).
  // Force the barrier back into a triggerable state: clear startedAtSync,
  // clear the propagation-kick bookkeeping, and turn it on. The next AtSync
  // round can then fire checkBarrier without bailing on !on or startedAtSync.
  void resetForRescale()
  {
    reset();
    on = true;
  }
#endif

#if CMK_SHRINK_EXPAND
  /** Undo the arrival of exactly the clients parked at `ceilingIter`, and
   *  resume them as though they had never called AtSync().
   *
   * Used when a rescale point is agreed at a later iteration than the one this
   * PE tentatively armed: elements parked at the old ceiling must be put back
   * to work to reach the agreed iteration. Exactly reverses what atBarrier()
   * did for them -- their round never started, because the barrier was turned
   * off for the duration, so there is no receiver state to unwind.
   *
   * Two filters, both load-bearing. A round already in flight is never
   * touched: its clients' arrivals were consumed when it fired, and resuming
   * them would race the round's own ResumeClients. And only clients whose
   * arrival is tagged with the old ceiling are rewound: a client that arrived
   * at its own regular load balancing boundary is a different round forming,
   * and pulling it out would leave that round waiting for it forever. */
  int rewindArrivedClients(int ceilingIter)
  {
    if (startedAtSync) return 0;
    std::vector<LBClient*> arrived;
    for (const auto& c : clients)
      if (c->epoch > curEpoch && c->arrivedAtIter == ceilingIter)
      {
        c->epoch--;
        atCount--;
        arrived.push_back(c);
      }
    for (const auto& c : arrived) c->fn();
    return (int)arrived.size();
  }
#endif

  void resumeClients();
  void resumeLateClients();

  bool hasReceivers() { return !receivers.empty(); };
};

#endif /* CKSYNCBARRIER_H */
