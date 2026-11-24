// Reductions whose contributors migrate in while ahead of the local PE.
//
// Each round, Main broadcasts migrate() and then, without waiting,
// hitsPerRound hit() broadcasts that each contribute to a sum reduction. A
// migrant can arrive having already contributed to several later reductions
// on its old PE; the new PE must still complete those rounds (issue #4145).
//
// Usage: ./reduction_migrate_ahead [rounds] [hits per round]

#include "reduction_migrate_ahead.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numRounds;
/*readonly*/ int hitsPerRound;
/*readonly*/ CProxy_Elem arrProxy;

static const int elemsPerPe = 2;

class Main : public CBase_Main {
  CProxy_Elem arr;
  int numElems;
  int round = 0;
  int reductionsThisRound = 0;

public:
  Main(CkArgMsg *m) {
    numRounds = m->argc > 1 ? atoi(m->argv[1]) : 50;
    hitsPerRound = m->argc > 2 ? atoi(m->argv[2]) : 4;
    delete m;
    mainProxy = thisProxy;
    numElems = elemsPerPe * CkNumPes();
    CkPrintf("reduction_migrate_ahead: %d elements, %d rounds, %d hits per round\n",
             numElems, numRounds, hitsPerRound);
    arr = CProxy_Elem::ckNew(numElems);
    arrProxy = arr;
    CcdCallFnAfter(watchdogTick, this, 2000);
    startRound();
  }

  // Watchdog: no progress in two consecutive 2 s ticks prints every
  // element's state and aborts, instead of holding a CI step for its
  // whole timeout.
  int wdRound = -1, wdRed = -1, wdStalls = 0;
  static void watchdogTick(void *arg, double) { ((Main *)arg)->watchdog(); }
  void watchdog() {
    if (round == wdRound && reductionsThisRound == wdRed) {
      if (++wdStalls == 1) {
        CkPrintf("reduction_migrate_ahead: no progress at round %d, reduction %d of %d\n",
                 round, reductionsThisRound, hitsPerRound);
        arr.report();
      } else {
        CkAbort("reduction_migrate_ahead: hung at round %d, reduction %d of %d", round, reductionsThisRound, hitsPerRound);
      }
    } else {
      wdRound = round; wdRed = reductionsThisRound; wdStalls = 0;
    }
    CcdCallFnAfter(watchdogTick, this, 2000);
  }

  void startRound() {
    reductionsThisRound = 0;
    arr.migrate(round);
    for (int h = 0; h < hitsPerRound; h++) arr.hit(round);
  }

  void hitDone(int n) {
    if (n != numElems)
      CkAbort("reduction_migrate_ahead: a reduction counted %d elements, expected %d", n, numElems);
    if (++reductionsThisRound < hitsPerRound) return;
    if (++round < numRounds) {
      startRound();
    } else {
      CkPrintf("reduction_migrate_ahead: %d reductions completed\n", numRounds * hitsPerRound);
      CkPrintf("All tests passed\n");
      CkExit();
    }
  }
};

class Elem : public CBase_Elem {
  int hits = 0;
  int migrations = 0;

public:
  Elem() {}
  Elem(CkMigrateMessage *m) : CBase_Elem(m) {}

  void pup(PUP::er &p) {
    p | hits;
    p | migrations;
  }

  void migrate(int round) {
    if (CkNumPes() == 1) return;
    migrations++;
    migrateMe((CkMyPe() + 1) % CkNumPes());
  }

  void report() {
    CkPrintf("[%d] elem %d: hits %d migrations %d\n", CkMyPe(), thisIndex, hits, migrations);
  }

  void hit(int round) {
    hits++;
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, hitDone), mainProxy));
  }
};

#include "reduction_migrate_ahead.def.h"
