// objid_boundmany: many arrays bound to one array, created from several PEs at
// once, migrate together. Each PE registers the bound arrays with the location
// manager in the order their creations reach it, which differs between PEs; a
// migrating element's data must still land in its own array. Each element of a
// bound array carries its array's number and checks it after every move, and
// checks that its partner in A came along. A round that makes no progress for
// 30 s fails the test (an element whose data went to another array never
// contributes).
//
// With -ckpt-at R the run checkpoints to log/ before round R and exits; a run
// with +restart log finishes the rounds. The restoring PE must put each
// element's data in its own array, whatever order the writing PE had.
#include "boundmany.decl.h"
#include <vector>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Creator creators;
/* readonly */ CProxy_A aProxy;
/* readonly */ int numArrays;
/* readonly */ int numElements;
/* readonly */ int rounds;

class Main : public CBase_Main {
  std::vector<CkArrayID> bound;
  int round = 0;
  int arraysChecked = 0;
  int ckptAt = -1;
  bool restarted = false;
  long progress = 0, progressSeen = -1;

  static void watchdog(void* self, double) {
    Main* m = static_cast<Main*>(self);
    if (m->progress == m->progressSeen) {
      CkPrintf("objid_boundmany FAIL: no progress for 30 s in round %d (%d of %d arrays "
               "checked, %d of %d created)\n", m->round, m->arraysChecked, numArrays,
               (int)m->bound.size(), numArrays);
      CkAbort("bound arrays stalled");
    }
    m->progressSeen = m->progress;
    CcdCallFnAfter(watchdog, self, 30000);
  }

public:
  Main(CkArgMsg* m) {
    numArrays = 64;
    rounds = 7;
    CmiGetArgInt(m->argv, "-a", &numArrays);
    CmiGetArgInt(m->argv, "-r", &rounds);
    CmiGetArgInt(m->argv, "-ckpt-at", &ckptAt);
    delete m;
    numElements = 4 * CkNumPes();
    mainProxy = thisProxy;
    aProxy = CProxy_A::ckNew(numElements);
    creators = CProxy_Creator::ckNew();
    // Each PE asks for some of the bound arrays.
    for (int k = 0; k < numArrays; k++) creators[k % CkNumPes()].make(k);
    CcdCallFnAfter(watchdog, this, 30000);
  }

  Main(CkMigrateMessage* m) : CBase_Main(m), restarted(true) {}

  void pup(PUP::er& p) {
    p | bound;
    p | round;
    p | arraysChecked;
    p | ckptAt;
    p | progress;
  }

  void made(CkArrayCreatedMsg* m) {
    bound.push_back(m->aid);
    delete m;
    progress++;
    if ((int)bound.size() == numArrays) next();
  }

  void next() {
    if (round == ckptAt && !restarted) {
      CkStartCheckpoint("log", CkCallback(CkIndex_Main::resumed(), thisProxy));
      return;
    }
    if (round == rounds) {
      CkPrintf("objid_boundmany PASS: %d bound arrays, %d elements each, %d rounds, %d PEs\n",
               numArrays, numElements, rounds, CkNumPes());
      CkExit();
      return;
    }
    for (auto& aid : bound) CProxy_B(aid).step(round, aid);
    round++;
  }

  void resumed() {
    if (!restarted) {
      CkPrintf("objid_boundmany: checkpoint written before round %d on %d PEs\n", round,
               CkNumPes());
      CkExit();
      return;
    }
    CkPrintf("objid_boundmany: restarted on %d PEs before round %d\n", CkNumPes(), round);
    ckptAt = -1;
    progressSeen = -1;
    CcdCallFnAfter(watchdog, this, 30000);
    next();
  }

  // One reduction per bound array.
  void checked(int n) {
    CkEnforce(n == numElements);
    progress++;
    if (++arraysChecked < numArrays) return;
    arraysChecked = 0;
    next();
  }
};

class Creator : public CBase_Creator {
public:
  Creator() {}
  Creator(CkMigrateMessage* m) : CBase_Creator(m) {}
  void make(int k) {
    CkArrayOptions opts(numElements);
    opts.bindTo(aProxy);
    CProxy_B::ckNew(k, opts, CkCallback(CkIndex_Main::made(nullptr), mainProxy));
  }
};

class A : public CBase_A {
public:
  A() {}
  A(CkMigrateMessage*) {}
};

class B : public CBase_B {
  int tag;
  int lastRound = -1;
  CkArrayID myArray;

public:
  B(int tag_) : tag(tag_), myArray(thisArrayID) {}
  B(CkMigrateMessage*) {}

  void pup(PUP::er& p) override {
    p | tag;
    p | lastRound;
    p | myArray;
  }

  void ckJustMigrated() override {
    CBase_B::ckJustMigrated();
    if (!(myArray == thisArrayID) || aProxy[thisIndex].ckLocal() == nullptr) {
      CkPrintf("objid_boundmany FAIL: element %d of bound array %d arrived on PE %d in "
               "array group %d (its data says group %d), partner in A %s\n",
               thisIndex, tag, CkMyPe(), ((CkGroupID)thisArrayID).idx,
               ((CkGroupID)myArray).idx,
               aProxy[thisIndex].ckLocal() ? "here" : "missing");
      CkAbort("bound element landed in the wrong array");
    }
  }

  // Every array's element thisIndex moves to the same next PE; the last
  // array's step moves the whole bound set. Round 0 does not move: once its
  // reductions are done, every bound array's manager exists on every PE, as
  // bound arrays require before their elements migrate.
  // sentTo is the array the broadcast went to. The element's own fields come
  // from its data, so an element holding another array's data names that
  // array in all of them.
  void step(int round, CkArrayID sentTo) {
    CkEnforce(lastRound == round - 1);
    if (!(myArray == sentTo) || !(thisArrayID == sentTo) ||
        aProxy[thisIndex].ckLocal() == nullptr) {
      CkPrintf("objid_boundmany FAIL: element %d of bound array %d on PE %d in round %d "
               "received a broadcast to array group %d but holds the data of group %d, "
               "partner in A %s\n",
               thisIndex, tag, CkMyPe(), round, ((CkGroupID)sentTo).idx,
               ((CkGroupID)myArray).idx, aProxy[thisIndex].ckLocal() ? "here" : "missing");
      CkAbort("bound element holds another array's data");
    }
    lastRound = round;
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, checked), mainProxy));
    if (round > 0 && tag == numArrays - 1) migrateMe((CkMyPe() + 1) % CkNumPes());
  }
};

#include "boundmany.def.h"
