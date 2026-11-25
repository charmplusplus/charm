// Every element contributes and then calls migrateMe, and the reduction starts
// a checkpoint at once, while elements are still migrating. A restart checks
// that all elements came back.
#include "ckpt_inflight.decl.h"

/* readonly */ CProxy_Main mainProxy;
static const int numElements = 64;

class Main : public CBase_Main {
  CProxy_E arr;
  bool restarted = false;

public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    arr = CProxy_E::ckNew(numElements);
    arr.hop();
  }
  Main(CkMigrateMessage* m) : CBase_Main(m), restarted(true) {}
  void pup(PUP::er& p) { p | arr; }

  void moved(int n) {
    CkEnforce(n == numElements);
    if (!restarted) {
      CkStartCheckpoint("log", CkCallback(CkIndex_Main::written(), thisProxy));
      return;
    }
    CkPrintf("ckpt_inflight PASS: %d elements after the restart on %d PEs\n", n, CkNumPes());
    CkExit();
  }

  // Called when the checkpoint is written, and after a restart.
  void written() {
    if (restarted) arr.hop();
    else CkExit();
  }
};

class E : public CBase_E {
public:
  E() {}
  E(CkMigrateMessage* m) : CBase_E(m) {}
  void hop() {
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, moved), mainProxy));
    migrateMe((CkMyPe() + 1) % CkNumPes());
  }
};

#include "ckpt_inflight.def.h"
