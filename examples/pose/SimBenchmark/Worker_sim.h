#ifndef Worker_H
#define Worker_H
#include "pose.h"
#include "Worker.decl.h"

#define MIX_MS 0
#define SMALL 1
#define MEDIUM 2
#define LARGE 3
#define SM_MSG_SZ 10
#define MD_MSG_SZ 100
#define LG_MSG_SZ 1000
#define MIX_GS 0
#define FINE 1
#define MEDIUM_GS 2
#define COARSE 3
#define FINE_GRAIN   0.000010
#define MEDIUM_GRAIN 0.001000
#define COARSE_GRAIN 0.010000
#define RANDOM 0
#define IMBALANCED 1
#define UNIFORM 2
#define SPARSE 0
#define HEAVY 1
#define FULL 2
class WorkerData : public eventMsg {
 public:
  int numObjs, numMsgs, msgSize, locality, density, grainSize,     msgsPerWork;
  double granularity;
  WorkerData& operator=(const WorkerData& obj) {
    int i;
    eventMsg::operator=(obj);
    numObjs = obj.numObjs;
    numMsgs = obj.numMsgs;
    msgSize = obj.msgSize;
    locality = obj.locality;
    grainSize = obj.grainSize;
    granularity = obj.granularity;
    density = obj.density;
    msgsPerWork = obj.msgsPerWork;
    return *this;
  }
};

class SmallWorkMsg : public eventMsg {
 public:
  int data[SM_MSG_SZ];
  SmallWorkMsg& operator=(const SmallWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<SM_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class MediumWorkMsg : public eventMsg {
 public:
  int data[MD_MSG_SZ];
  MediumWorkMsg& operator=(const MediumWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<MD_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class LargeWorkMsg : public eventMsg {
 public:
  int data[LG_MSG_SZ];
  LargeWorkMsg& operator=(const LargeWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<LG_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class worker : public sim {
 private:
   void ResolveFn(int fnIdx, void *msg);
   void ResolveCommitFn(int fnIdx, void *msg);
 public:
   worker(CkMigrateMessage *) {};
   void pup(PUP::er &p);
    worker(WorkerData *);
      void workSmall(SmallWorkMsg *);
    void workMedium(MediumWorkMsg *);
    void workLarge(LargeWorkMsg *);
};

class state_worker : public chpt<state_worker> {
  friend class worker;
  int numObjs, numMsgs, msgSize, locality, grainSize, density, msgsPerWork,    sent, totalObjs, elapseTime, elapseRem, neighbor;
  double granularity, localDensity;
  int data[100];
 public:
  state_worker(sim *p, strat *s) { parent = p; myStrat = s; }
  state_worker();
  state_worker(WorkerData *m); 
  ~state_worker() { }
  state_worker& operator=(const state_worker& obj);
  void pup(PUP::er &p) { 
#ifndef SEQUENTIAL_POSE
    chpt<state_worker>::pup(p); 
#endif
    p(numObjs); p(numMsgs); p(msgSize); p(density); p(localDensity);
    p(locality); p(grainSize); p(granularity); p(msgsPerWork); p(sent);
    p(totalObjs); p(elapseTime); p(elapseRem); p(neighbor);
    p(data, 100);
  }
  void cpPup(PUP::er &p) {
    p(numObjs); p(numMsgs); p(msgSize); p(density); p(localDensity);
    p(locality); p(grainSize); p(granularity); p(msgsPerWork); p(sent);
    p(totalObjs); p(elapseTime); p(elapseRem); p(neighbor);
    p(data, 100);
  }
  void dump() {
#ifndef SEQUENTIAL_POSE
    chpt<state_worker>::dump();
#endif
    CkPrintf("[state_worker: ");
  }
  void doWork();
  // Event methods
  void workSmall(SmallWorkMsg *m);
  void workSmall_anti(SmallWorkMsg *m);
  void workSmall_commit(SmallWorkMsg *m);
  void workMedium(MediumWorkMsg *m);
  void workMedium_anti(MediumWorkMsg *m);
  void workMedium_commit(MediumWorkMsg *m);
  void workLarge(LargeWorkMsg *m);
  void workLarge_anti(LargeWorkMsg *m);
  void workLarge_commit(LargeWorkMsg *m);
};

#endif
