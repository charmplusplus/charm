// objid_bindasync: an asynchronous creation request for an array bound to
// another (bindTo) succeeds when it reaches PE 0 before the bound-to location
// manager is built there (charm #4174).
// Each round PE 1 creates a custom map M and, asynchronously, array A on M; then,
// once A's branch exists on PE 1, array B bound to A. A's location manager on
// PE 0 is held until M's creation reaches PE 0. PE 0 stalls while PE 1 floods it
// with no-op messages, so under +randomized_msgq M's creation is taken at a
// random point among them and B's request can come first. B's elements must sit
// with A's. Needs 2 or more PEs.
#include "bindasync.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Driver driver;
/* readonly */ int rounds;
/* readonly */ int floodCount;

static int numElements() { return 2 * CkNumPes(); }

class Main : public CBase_Main {
  int round = 0;
  CProxy_B b;

public:
  Main(CkArgMsg* m) {
    rounds = 10;
    floodCount = 400;
    CmiGetArgInt(m->argv, "-r", &rounds);
    CmiGetArgInt(m->argv, "-f", &floodCount);
    delete m;
    if (CkNumPes() < 2) {
      CkPrintf("objid_bindasync: needs 2 or more PEs; nothing to check\n");
      CkExit();
      return;
    }
    mainProxy = thisProxy;
    driver = CProxy_Driver::ckNew();
    driver[1].begin(round);
  }

  void bMade(CkArrayCreatedMsg* m) {
    b = CProxy_B(m->aid);
    delete m;
    b.check();
  }

  // Every element of B found its partner in A on its own PE.
  void checked(int n) {
    CkEnforce(n == numElements());
    if (++round == rounds) {
      CkPrintf("objid_bindasync PASS: %d rounds, %d PEs, every bound array created and "
               "colocated\n", rounds, CkNumPes());
      CkExit();
      return;
    }
    driver[1].begin(round);
  }
};

class Driver : public CBase_Driver {
  CkArrayID aid;

public:
  Driver() {}

  // On PE 1.
  void begin(int round) {
    thisProxy[0].stall();
    for (int i = 0; i < floodCount; i++) thisProxy[0].ping();
    CProxy_Map m = CProxy_Map::ckNew();
    CkArrayOptions opts(numElements());
    opts.setMap(m);
    CProxy_A::ckNew(opts, CkCallback(CkIndex_Driver::aMade(nullptr), thisProxy[1]));
  }

  // On PE 0: spin so that the messages from PE 1 queue up together.
  void stall() {
    const double t0 = CkWallTimer();
    while (CkWallTimer() - t0 < 0.005) {}
  }

  void ping() {}

  void aMade(CkArrayCreatedMsg* m) {
    aid = m->aid;
    delete m;
    tryBind();
  }

  // bindTo reads A's branch on this PE, which may not be built yet.
  void tryBind() {
    if (CProxy_CkArray(aid).ckLocalBranch() == nullptr) {
      thisProxy[CkMyPe()].tryBind();
      return;
    }
    CkArrayOptions opts(numElements());
    opts.bindTo(aid);
    CProxy_B::ckNew(aid, opts, CkCallback(CkIndex_Main::bMade(nullptr), mainProxy));
  }
};

class Map : public CBase_Map {
public:
  Map() {}
  int procNum(int, const CkArrayIndex& idx) override {
    return (idx.data()[0] * 7 + 3) % CkNumPes();
  }
};

class A : public CBase_A {
public:
  A() {}
  A(CkMigrateMessage*) {}
};

class B : public CBase_B {
  CkArrayID aid;

public:
  B(CkArrayID aid_) : aid(aid_) {}
  B(CkMigrateMessage*) {}

  // A's manager on this PE can be built after B's.
  void check() {
    if (CProxy_CkArray(aid).ckLocalBranch() == nullptr) {
      thisProxy[thisIndex].check();
      return;
    }
    if (CProxy_A(aid)[thisIndex].ckLocal() == nullptr) {
      CkPrintf("objid_bindasync FAIL: B[%d] on PE %d has no A[%d] there\n", thisIndex,
               CkMyPe(), thisIndex);
      CkAbort("bound array not colocated");
    }
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, checked), mainProxy));
  }
};

#include "bindasync.def.h"
