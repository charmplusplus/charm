// Reductions in flight across AtSync load balancing.
//
// Every step, every element contributes to nRed reductions and then calls
// AtSync() as the last action of the same entry method, so the element
// migrates (under a balancer that moves it) while its contributions are
// still being combined. By default each result is broadcast back to the
// array, as an allreduce: every element checks the nRed results against the
// closed form and then contributes to a step-done reduction to the
// mainchare, which starts the next step. With -m the results go to the
// mainchare instead, which checks them and starts the next step.
// The default is the AMPI pattern "post several MPI_Iallreduce,
// AMPI_Migrate, MPI_Waitall" without AMPI (charm #4072).
//
// With -d (drained) an element calls AtSync() only after the step's results
// have arrived, and contributes to the step-done reduction from
// ResumeFromSync, so no broadcast to the array is in flight while elements
// move. That is the promise an array created with -s makes; -s without -d
// breaks it and hangs (doc/internals/array-broadcast-migration.md).
//
// Each contribution carries its reduction index in bits 40 and up, and each
// result is checked against the reduction it decodes to, so the results of
// one step may arrive in any order. With -o the results must also arrive in
// reduction order: the array delivers its broadcasts to every element in one
// consistent order, and the reductions complete on PE 0, which also stamps
// that order, so here it is the completion order.
//
// A stall is reported by a watchdog: after -w seconds with no reduction
// completing, every element prints its state and the run aborts.
//
// Flags: -n <elements, default 8>  -i <steps, default 20>
//        -r <reductions per step, default 3>
//        -m (reduction results to the mainchare, not broadcast to the array)
//        -b (bound arrays: a second array owns usesAtSync and calls AtSync;
//            the contributing element is bound to it and migrates with it,
//            which is how AMPI's ampi array rides on TCharm)
//        -k (results broadcast to a [nokeep] entry, as AMPI's irednResult)
//        -s (arrays created with setStaticInsertion(true) and
//            setAnytimeMigration(false), as TCharm and AMPI create theirs:
//            broadcasts are not stored for migrating elements)
//        -d (AtSync only after the step's results arrived; see above)
//        -o (require the results of a step in reduction order)
//        -M (require at least one migration; use with a balancer and >1 PE)
//        -A (no AtSync: reductions only, checks the test's own protocol)
//        -w <watchdog stall threshold in seconds, default 10; 0 disables>
//        -v (verbose)
//
// Run: ./redmig +pe 2 +balancer RandCentLB -M

#include "redmig.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem elemProxy;
/*readonly*/ int nElems;
/*readonly*/ int nSteps;
/*readonly*/ int nRed;
/*readonly*/ int noSync;
/*readonly*/ int toMain;
/*readonly*/ int useBound;
/*readonly*/ int noKeep;
/*readonly*/ int staticIns;
/*readonly*/ int drained;
/*readonly*/ int inOrder;
/*readonly*/ CProxy_Sync syncProxy;
/*readonly*/ int verbose;

static inline unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// The value element i contributes to reduction k of step s.
static inline long long redVal(int i, int s, int k) {
  return (long long)(mix64(((unsigned long long)(unsigned)s << 40) |
                           ((unsigned long long)(unsigned)k << 20) |
                           (unsigned)i) % 1000003ULL);
}

static const long long kShift = 40;
static const long long kMask = (1LL << kShift) - 1;

// Splits a summed result into its reduction index and the sum of the values;
// returns false if the sum is not that of reduction k of step s.
static bool checkResult(long long sum, int s, int* kOut) {
  int k = (int)((sum >> kShift) / nElems);
  *kOut = k;
  if (k < 0 || k >= nRed || (sum >> kShift) != (long long)k * nElems) return false;
  long long expected = 0;
  for (int i = 0; i < nElems; i++) expected += redVal(i, s, k);
  return (sum & kMask) == expected;
}

static void watchdogFire(void* arg, double t);
static void abortFire(void* arg, double t) {
  fflush(stdout);
  CkAbort("redmig: hung (state dumped above)");
}

class Main : public CBase_Main {
  int step = 0;
  int redIdx = 0;        // results of this step received so far
  unsigned long long redMask = 0;  // which reductions of this step arrived
  double t0;
  double lastAdvance = 0;
  bool dumped = false;
  int wdSecs = 10;
  int requireMigration = 0;

public:
  Main(CkArgMsg* m) {
    nElems = 8; nSteps = 20; nRed = 3; noSync = 0; toMain = 0; useBound = 0;
    noKeep = 0; staticIns = 0; drained = 0; inOrder = 0;
    verbose = 0;
    CmiGetArgInt(m->argv, "-n", &nElems);
    CmiGetArgInt(m->argv, "-i", &nSteps);
    CmiGetArgInt(m->argv, "-r", &nRed);
    CmiGetArgInt(m->argv, "-w", &wdSecs);
    if (CmiGetArgFlag(m->argv, "-v")) verbose = 1;
    if (CmiGetArgFlag(m->argv, "-A")) noSync = 1;
    if (CmiGetArgFlag(m->argv, "-m")) toMain = 1;
    if (CmiGetArgFlag(m->argv, "-b")) useBound = 1;
    if (CmiGetArgFlag(m->argv, "-k")) noKeep = 1;
    if (CmiGetArgFlag(m->argv, "-s")) staticIns = 1;
    if (CmiGetArgFlag(m->argv, "-d")) drained = 1;
    if (CmiGetArgFlag(m->argv, "-o")) inOrder = 1;
    if (CmiGetArgFlag(m->argv, "-M")) requireMigration = 1;
    delete m;
    CkEnforce(nElems >= 1 && nSteps >= 1 && nRed >= 1);
    CkEnforce(!(drained && toMain));  // -d needs the results at the array
    CkEnforce(nRed <= 64 && nElems <= 100000);  // the result encoding
    CkPrintf("redmig: %d PEs, %d processes, %d elements, %d steps, "
             "%d reductions per step, results to %s, AtSync %s%s%s\n",
             CkNumPes(), CkNumNodes(), nElems, nSteps, nRed,
             toMain ? "the mainchare"
                    : noKeep ? "the array (nokeep broadcast)" : "the array (broadcast)",
             noSync ? "off" : drained ? "after the step's results arrived"
                                      : "after the contributions",
             useBound ? " on a bound array" : "",
             staticIns ? ", static insertion" : "");
    mainProxy = thisProxy;
    CkArrayOptions opts(nElems);
    if (staticIns) { opts.setStaticInsertion(true); opts.setAnytimeMigration(false); }
    if (useBound) {
      CkArrayOptions sopts(nElems);
      if (staticIns) { sopts.setStaticInsertion(true); sopts.setAnytimeMigration(false); }
      syncProxy = CProxy_Sync::ckNew(sopts);
      opts.bindTo(syncProxy);
    }
    elemProxy = CProxy_Elem::ckNew(opts);
    t0 = CkWallTimer();
    lastAdvance = t0;
    if (wdSecs > 0) CcdCallFnAfter(watchdogFire, NULL, 2000);
    elemProxy.nextStep(0);
  }

  void checkProgress() {
    if (!dumped && CkWallTimer() - lastAdvance > (double)wdSecs) {
      dumped = true;
      if (toMain)
        CkPrintf("redmig WATCHDOG: no progress for %d s; main at step %d "
                 "waiting for reduction %d of %d. Element dumps:\n",
                 wdSecs, step, redIdx, nRed);
      else
        CkPrintf("redmig WATCHDOG: no progress for %d s; main at step %d "
                 "waiting for the step-done reduction. Element dumps:\n",
                 wdSecs, step);
      elemProxy.dumpState();
      CcdCallFnAfter(abortFire, NULL, 3000);
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void redDone(long long sum) {
    lastAdvance = CkWallTimer();
    int k;
    if (!checkResult(sum, step, &k))
      CkPrintf("redmig: step %d result %lld MISMATCH (decodes to reduction %d)\n",
               step, sum, k);
    CkEnforce(checkResult(sum, step, &k));
    CkEnforce(!((redMask >> k) & 1));  // each result once
    if (inOrder && k != redIdx)
      CkPrintf("redmig: step %d reduction %d arrived as result %d OUT OF ORDER\n", step, k, redIdx);
    CkEnforce(!inOrder || k == redIdx);
    redMask |= 1ULL << k;
    if (verbose) CkPrintf("redmig: step %d reduction %d ok\n", step, k);
    if (++redIdx < nRed) return;
    redIdx = 0;
    redMask = 0;
    advance();
  }

  // Every element received this step's nRed broadcast results (with -d: and
  // resumed from the balancing step that followed).
  void stepDone(int n) {
    lastAdvance = CkWallTimer();
    CkEnforce(n == nElems);
    if (verbose) CkPrintf("redmig: step %d done at every element\n", step);
    advance();
  }

  void advance() {
    step++;
    if (step < nSteps) elemProxy.nextStep(step);
    else elemProxy.finish();
  }

  void finalStats(long long totalMigrations) {
    CkPrintf("redmig: %d steps, %lld migrations, %.3f s\n",
             nSteps, totalMigrations, CkWallTimer() - t0);
    if (requireMigration && CkNumPes() > 1) CkEnforce(totalMigrations > 0);
    CkPrintf("redmig PASS\n");
    CkExit();
  }
};

class Elem;

// The AtSync owner in bound mode. Its Elem is co-located, so each forwards
// to the other with ckLocal.
class Sync : public CBase_Sync {
public:
  Sync() { usesAtSync = true; }
  Sync(CkMigrateMessage* m) : CBase_Sync(m) {}
  void pup(PUP::er& p) {}
  void ResumeFromSync();
};

class Elem : public CBase_Elem {
  int curStep;          // last step started; -1 before step 0
  int pendingStep;      // step received while between AtSync and ResumeFromSync
  bool inSync;          // AtSync called, ResumeFromSync not yet
  int gotRes;           // broadcast results received for curStep
  unsigned long long gotMask;  // which reductions of curStep arrived
  long long migrations;
  int resumes;

  void doStep(int s) {
    curStep = s;
    gotRes = 0;
    gotMask = 0;
    CkCallback cb = toMain
        ? CkCallback(CkReductionTarget(Main, redDone), mainProxy)
        : noKeep ? CkCallback(CkIndex_Elem::redResultNoKeep(0), elemProxy)
                 : CkCallback(CkReductionTarget(Elem, redResult), elemProxy);
    for (int k = 0; k < nRed; k++) {
      long long v = redVal(thisIndex, s, k) + ((long long)k << kShift);
      contribute(sizeof(long long), &v, CkReduction::sum_long_long, cb);
    }
    if (noSync || drained) return;
    callAtSync();  // last action; nRed contributions are in flight
  }

  void callAtSync() {
    inSync = true;
    if (useBound) {
      Sync* owner = syncProxy[thisIndex].ckLocal();
      CkEnforce(owner != NULL);
      owner->AtSync();
    } else {
      AtSync();
    }
  }

  void stepDone() {
    int one = 1;
    contribute(sizeof(int), &one, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, stepDone), mainProxy));
  }

public:
  Elem() : curStep(-1), pendingStep(-1), inSync(false), gotRes(0), gotMask(0),
           migrations(0), resumes(0) {
    if (!useBound) usesAtSync = true;
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p) {
    p | curStep; p | pendingStep; p | inSync; p | gotRes; p | gotMask;
    p | migrations;
    p | resumes;
    if (p.isUnpacking()) migrations++;
  }

  void nextStep(int s) {
    CkEnforce(s == curStep + 1);
    if (inSync) { pendingStep = s; return; }
    doStep(s);
  }

  // A result of curStep, in any order, never before this element contributed
  // to that step.
  void redResult(long long sum) {
    CkEnforce(gotRes < nRed);
    int k;
    if (!checkResult(sum, curStep, &k))
      CkPrintf("redmig: elem %d step %d result %lld MISMATCH (decodes to "
               "reduction %d)\n", thisIndex, curStep, sum, k);
    CkEnforce(checkResult(sum, curStep, &k));
    CkEnforce(!((gotMask >> k) & 1));  // each result once
    gotMask |= 1ULL << k;
    if (inOrder && k != gotRes)
      CkPrintf("redmig: elem %d PE %d step %d reduction %d arrived as result %d OUT OF ORDER\n",
               thisIndex, CkMyPe(), curStep, k, gotRes);
    CkEnforce(!inOrder || k == gotRes);
    if (++gotRes < nRed) return;
    if (drained && !noSync) callAtSync();  // every broadcast of this step has arrived here
    else stepDone();
  }

  void redResultNoKeep(CkReductionMsg* m) {
    redResult(*(long long*)m->getData());  // [nokeep]: the message is not ours
  }

  void ResumeFromSync() {
    CkEnforce(inSync);
    inSync = false;
    resumes++;
    if (drained) { stepDone(); return; }
    if (pendingStep >= 0) {
      int s = pendingStep;
      pendingStep = -1;
      doStep(s);
    }
  }

  void dumpState() {
    CkPrintf("redmig DUMP: elem %d on PE %d: curStep %d gotRes %d (mask %llx) "
             "inSync %d pendingStep %d resumes %d migrations %lld\n",
             thisIndex, CkMyPe(), curStep, gotRes, gotMask, (int)inSync,
             pendingStep, resumes, migrations);
  }

  void finish() {
    CkEnforce(curStep == nSteps - 1);
    contribute(sizeof(long long), &migrations, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, finalStats), mainProxy));
  }
};

void Sync::ResumeFromSync() {
  Elem* e = elemProxy[thisIndex].ckLocal();
  CkEnforce(e != NULL);
  e->ResumeFromSync();
}

static void watchdogFire(void* arg, double t) { mainProxy.checkProgress(); }

#include "redmig.def.h"
