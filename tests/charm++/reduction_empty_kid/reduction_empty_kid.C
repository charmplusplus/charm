// A PE that empties while holding contributions its elements left behind.
//
// Usage: ./reduction_empty_kid (3 or more PEs)
//
// PE 1 holds elements A and B. A contributes to four reductions at once and
// migrates to PE 2, then B migrates to PE 2 and contributes there. PE 1 is
// left with no element and A's four contributions. It tells its parent it is
// empty, then completes the four reductions from what A left behind and sends
// real results up. A fifth reduction needs PE 1's empty message, so the root
// must still treat PE 1 as empty.
//
// Needs at least three PEs; with the block map, PE 1 holds elements 2 and 3.

#include "reduction_empty_kid.decl.h"

/*readonly*/ CProxy_Main mainProxy;

static const int kBurst = 4;

class Main : public CBase_Main {
  CProxy_Elem arr;
  int numElems;
  int completed = 0;
  int phase = 0;

public:
  Main(CkArgMsg *m) {
    delete m;
    if (CkNumPes() < 3) CkAbort("reduction_empty_kid needs at least 3 PEs");
    mainProxy = thisProxy;
    numElems = 2 * CkNumPes();  // block map: two elements per PE
    arr = CProxy_Elem::ckNew(numElems);
    arr[2].burst(kBurst);  // A: contributes to reductions 0..3, leaves PE 1, then sends B away
    for (int e = 0; e < numElems; e++)
      if (e != 2 && e != 3)
        for (int k = 0; k < kBurst; k++) arr[e].hit();
    CcdCallFnAfter(watchdogTick, this, 3000);
  }

  static void watchdogTick(void *arg, double) { ((Main *)arg)->watchdog(); }
  int wdCompleted = -1;
  void watchdog() {
    if (completed == wdCompleted)
      CkAbort("reduction_empty_kid: hung after %d reductions (phase %d)", completed, phase);
    wdCompleted = completed;
    CcdCallFnAfter(watchdogTick, this, 3000);
  }

  void done(int n) {
    if (n != numElems)
      CkAbort("reduction_empty_kid: reduction %d counted %d elements, expected %d", completed, n, numElems);
    completed++;
    if (completed == kBurst && phase == 0) {
      phase = 1;
      arr.hit();  // reduction 4: PE 1 has no element and must still report
    } else if (completed == kBurst + 1) {
      CkPrintf("All tests passed\n");
      CkExit();
    }
  }
};

class Elem : public CBase_Elem {
public:
  Elem() {}
  Elem(CkMigrateMessage *m) : CBase_Elem(m) {}
  void pup(PUP::er &p) {}

  void burst(int n) {
    for (int k = 0; k < n; k++) hit();
    migrateMe(2);            // queued behind this method: A leaves first,
    thisProxy[3].leave(2);   // then B
  }
  void leave(int toPe) { migrateMe(toPe); }
  void ckJustMigrated() override {
    CBase_Elem::ckJustMigrated();
    if (thisIndex == 3)  // B contributes to reductions 0..3 after leaving PE 1
      for (int k = 0; k < kBurst; k++) hit();
  }
  void hit() {
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

#include "reduction_empty_kid.def.h"
