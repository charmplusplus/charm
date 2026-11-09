// Elements move with migrateMe(), then the array runs AtSync load-balancing
// steps. The balancer must count only the moves it ordered (#4195): before
// that fix the migrateMe arrivals left TreeLB's per-step count stale, and the
// next step asserted (debug) or never resumed its elements (production).
//   which = 0: no migrateMe (control), 1: odd elements move, 2: all move
#include "migrate_then_atsync.decl.h"

CProxy_Main mainProxy;

class Main : public CBase_Main {
  CProxy_Elem arr;
  int which, period = 0;
  static constexpr int periods = 3;

 public:
  Main(CkArgMsg* m) {
    which = m->argc > 1 ? atoi(m->argv[1]) : 1;
    delete m;
    mainProxy = thisProxy;
    arr = CProxy_Elem::ckNew(4 * CkNumPes());
    arr.mig(which);
    CcdCallFnAfter(watchdog, this, 30000);
  }
  void migDone() { arr.dosync(); }
  void syncDone() {
    if (++period == periods) {
      CkPrintf("PASS which=%d\n", which);
      CkExit();
    } else
      arr.dosync();
  }
  static void watchdog(void* arg, double) {
    Main* m = (Main*)arg;
    CkAbort("migrate_then_atsync: hung in load balancing step %d of %d (which=%d)", m->period + 1,
            periods, m->which);
  }
};

class Elem : public CBase_Elem {
  bool moving = false;

 public:
  Elem() { usesAtSync = true; }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void pup(PUP::er& p) { p | moving; }

  void mig(int which) {
    if (which == 2 || (which == 1 && thisIndex % 2 == 1)) {
      moving = true;
      migrateMe((CkMyPe() + 1) % CkNumPes());
    } else
      contribute(CkCallback(CkReductionTarget(Main, migDone), mainProxy));
  }
  void ckJustMigrated() {
    CBase_Elem::ckJustMigrated();
    if (moving) {
      moving = false;
      contribute(CkCallback(CkReductionTarget(Main, migDone), mainProxy));
    }
  }
  void dosync() { AtSync(); }
  void ResumeFromSync() { contribute(CkCallback(CkReductionTarget(Main, syncDone), mainProxy)); }
};

#include "migrate_then_atsync.def.h"
