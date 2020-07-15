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

CkpvExtern(bool, cksyncbarrierInited);

class CkSyncBarrier : public CBase_CkSyncBarrier
{
private:
  std::list<LBClient*> clients;
  std::list<LBReceiver*> receivers;

  std::list<int> local_pes_to_notify;

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
    reset();
  }

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

  void propagate_atsync();
  void recvLbStart(int lb_step, int sourcenode, int pe);

  LDBarrierClient AddClient(Chare* chare, std::function<void()> fn);
  void RemoveClient(LDBarrierClient h);
  LDBarrierReceiver AddReceiver(std::function<void()> fn);
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
  void reset();

  void CallReceivers(void);
  void CheckBarrier(bool flood_atsync = false);
  void ResumeClients(void);
};

#endif /* CKSYNCBARRIER_H */
