// Anytime-migration stress test: neighbor exchange + broadcasts + reductions.
//
// Sibling of tests/charm++/anytime_migration (the bare #3660 ping reproducer).
// This one drives the full trio of mechanisms that must survive an element
// migrating at an arbitrary point via migrateMe():
//
//   - per-step LEFT/RIGHT neighbor exchange on a 1D ring (kNeighbor pattern),
//     every payload a deterministic function of (index, step) so the receiver
//     verifies exactness, not just arrival;
//   - a sum reduction per step whose value the mainchare checks against the
//     closed form;
//   - a broadcast per step (the next-step go-ahead), which each element
//     enforces arrives exactly once and in order.
//
// At every migPeriod-th step, migPerEvent pseudo-randomly chosen elements
// migrate to a pseudo-random other PE. The migrateMe() is issued as the LAST
// action of the entry method in which the element completes its contribution
// for the step (the manual's requirement), so the element leaves while its
// reduction contribution is still in flight -- the interesting redNo path.
//
// A neighbor message for step S+1 can legally arrive before the step-S+1
// broadcast (p2p is not ordered against broadcasts), so such messages are
// buffered; the buffer is pupped, so a migration with a buffered early
// message also tests state carry-over. Messages can never be LATE (the
// mainchare only advances after every element contributed), so lateness is
// enforced as an error.
//
// All checks use CkEnforce (CkAssert is a no-op in production builds).
// Expected failure modes of a rusty anytime-migration path: hang (lost
// forwarded message / lost broadcast to a migrant), abort on one of the
// enforces (duplicate or misordered delivery, wrong reduction sum), or crash.
//
// Flags: -n <elems, >=3, default 8>  -i <steps, default 60>
//        -m <migration period in steps, default 10; 0 disables migration>
//        -c <elements migrated per event, default 1>
//        -S <seed, default 42>  -v (verbose)
//
// Run (reconverse): single process  ./bcastred +pe 4
//   multi-process   lcrun -n 2 env DYLD_LIBRARY_PATH=<build>/lib ./bcastred +pe 4

#include "bcastred.decl.h"
#include "pup_stl.h"
#include <map>
#include <vector>
#include <utility>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem elemProxy;
/*readonly*/ int nElems;
/*readonly*/ int nSteps;
/*readonly*/ int migPeriod;
/*readonly*/ int migPerEvent;
/*readonly*/ int gSeed;
/*readonly*/ int verbose;

// splitmix64: deterministic, replicated on every PE -- no state to pup.
static inline unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// The value element i sends/contributes at step s. Small enough that
// nElems * nSteps * 1e6 stays far from long long overflow.
static inline long long stepVal(int i, int s) {
  return (long long)(mix64(((unsigned long long)(unsigned)s << 32) |
                           (unsigned)i) % 1000003ULL);
}

// Is element idx scheduled to migrate at step s? (Same schedule computed
// everywhere -- deterministic lockstep, nothing broadcast.)
static bool migratesAt(int s, int idx) {
  if (migPeriod <= 0 || s == 0 || s % migPeriod != 0) return false;
  for (int k = 0; k < migPerEvent; k++)
    if ((int)(mix64((unsigned long long)gSeed ^
                    ((unsigned long long)s * 131071ULL + k)) % nElems) == idx)
      return true;
  return false;
}

static void watchdogFire(void* arg, double t);
static void abortFire(void* arg, double t) {
  fflush(stdout);
  CkAbort("bcastred: hung (state dumped above)");
}

class Main : public CBase_Main {
  int step = 0;
  double t0;
  double lastAdvance = 0;
  bool dumped = false;

public:
  Main(CkArgMsg* m) {
    nElems = 8; nSteps = 60; migPeriod = 10; migPerEvent = 1;
    gSeed = 42; verbose = 0;
    CmiGetArgInt(m->argv, "-n", &nElems);
    CmiGetArgInt(m->argv, "-i", &nSteps);
    CmiGetArgInt(m->argv, "-m", &migPeriod);
    CmiGetArgInt(m->argv, "-c", &migPerEvent);
    CmiGetArgInt(m->argv, "-S", &gSeed);
    if (CmiGetArgFlag(m->argv, "-v")) verbose = 1;
    delete m;
    CkEnforce(nElems >= 3);  // ring with distinct left/right neighbors
    CkEnforce(nSteps >= 1);
    if (CkNumPes() < 2 && migPeriod > 0) {
      CkPrintf("bcastred: only 1 PE, migration disabled\n");
      migPeriod = 0;
    }
    CkPrintf("bcastred: %d PEs, %d processes, %d elements, %d steps, "
             "migration every %d steps x %d elements, seed %d\n",
             CkNumPes(), CkNumNodes(), nElems, nSteps, migPeriod,
             migPerEvent, gSeed);
    mainProxy = thisProxy;
    elemProxy = CProxy_Elem::ckNew(nElems);
    t0 = CkWallTimer();
    lastAdvance = t0;
    CcdCallFnAfter(watchdogFire, NULL, 2000);
    elemProxy.nextStep(0);
  }

  // Watchdog (Ccd timer -> entry message): if no reduction completed for
  // 5 s, ask every element to dump its protocol state -- distinguishes a
  // lost neighbor message (gotNbr<2) from a lost broadcast (curStep behind)
  // from a lost reduction contribution (all contributed, no stepDone).
  void checkProgress() {
    if (!dumped && CkWallTimer() - lastAdvance > 5.0) {
      dumped = true;
      CkPrintf("bcastred WATCHDOG: no progress for 5 s; main at step %d "
               "(waiting for that step's reduction). Element dumps:\n", step);
      elemProxy.dumpState();
      CcdCallFnAfter(abortFire, NULL, 3000);  // time for dumps, then abort
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void stepDone(long long sum) {
    lastAdvance = CkWallTimer();
    long long expected = 0;
    for (int i = 0; i < nElems; i++) expected += stepVal(i, step);
    if (sum != expected)
      CkPrintf("bcastred: step %d reduction MISMATCH got %lld expected %lld\n",
               step, sum, expected);
    CkEnforce(sum == expected);
    if (verbose) CkPrintf("bcastred: step %d ok (sum %lld)\n", step, sum);
    step++;
    if (step < nSteps) elemProxy.nextStep(step);
    else               elemProxy.finish();
  }

  void finalStats(long long totalMigrations) {
    // Count the migrations the schedule promises (duplicates within an
    // event migrate once, so count distinct chosen elements per event).
    long long promised = 0;
    for (int s = 1; s < nSteps; s++) {
      if (migPeriod <= 0 || s % migPeriod != 0) continue;
      for (int i = 0; i < nElems; i++) if (migratesAt(s, i)) promised++;
    }
    if (totalMigrations != promised)
      CkPrintf("bcastred: migration count MISMATCH got %lld expected %lld\n",
               totalMigrations, promised);
    CkEnforce(totalMigrations == promised);
    CkPrintf("bcastred PASS: %d steps, %lld migrations, %.3f s\n",
             nSteps, totalMigrations, CkWallTimer() - t0);
    CkExit();
  }
};

class Elem : public CBase_Elem {
  int curStep;          // last broadcast step received; -1 before step 0
  int bcastCount;       // broadcasts received; enforced == curStep+1
  int gotNbr;           // neighbor messages consumed for curStep
  bool contributed;     // completion guard (per-step, at most once)
  long long migrations; // times this element migrated
  long long checksum;   // sum of every value seen (own + neighbors)
  // step -> messages that arrived before that step's broadcast
  std::map<int, std::vector<std::pair<int, long long> > > early;

  int left()  const { return (thisIndex + nElems - 1) % nElems; }
  int right() const { return (thisIndex + 1) % nElems; }

  void consumeNbr(int fromIdx, int step, long long val) {
    CkEnforce(fromIdx == left() || fromIdx == right());
    if (val != stepVal(fromIdx, step))
      CkPrintf("bcastred: elem %d step %d bad value from %d: got %lld "
               "expected %lld\n", thisIndex, step, fromIdx, val,
               stepVal(fromIdx, step));
    CkEnforce(val == stepVal(fromIdx, step));
    gotNbr++;
    CkEnforce(gotNbr <= 2);
    checksum += val;
  }

  // If this step's work is complete, contribute -- and then, possibly,
  // migrate. Callers must make this their LAST action: migrateMe() must end
  // the entry method.
  void maybeComplete() {
    if (curStep < 0 || contributed || gotNbr < 2) return;
    contributed = true;
    long long v = stepVal(thisIndex, curStep);
    checksum += v;
    contribute(sizeof(long long), &v, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, stepDone), mainProxy));
    if (migratesAt(curStep, thisIndex) && CkNumPes() > 1) {
      int hop = 1 + (int)(mix64((unsigned long long)gSeed * 0x51ED2701ULL +
                                (unsigned long long)(unsigned)curStep * nElems +
                                thisIndex) % (CkNumPes() - 1));
      int dest = (CkMyPe() + hop) % CkNumPes();
      migrations++;
      CkPrintf("bcastred: step %d elem %d migrating PE %d -> %d (#%lld)\n",
               curStep, thisIndex, CkMyPe(), dest, migrations);
      migrateMe(dest);  // last action; reduction contribution is in flight
    }
  }

public:
  Elem() : curStep(-1), bcastCount(0), gotNbr(0), contributed(false),
           migrations(0), checksum(0) {}
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p) {
    p | curStep; p | bcastCount; p | gotNbr; p | contributed;
    p | migrations; p | checksum; p | early;
  }

  void nextStep(int step) {
    if (step != curStep + 1)
      CkPrintf("bcastred: elem %d broadcast misorder: got step %d at "
               "curStep %d\n", thisIndex, step, curStep);
    CkEnforce(step == curStep + 1);   // exactly once, in order
    curStep = step;
    bcastCount++;
    CkEnforce(bcastCount == step + 1);
    gotNbr = 0;
    contributed = false;
    elemProxy[left()].neighborVal(thisIndex, step, stepVal(thisIndex, step));
    elemProxy[right()].neighborVal(thisIndex, step, stepVal(thisIndex, step));
    std::map<int, std::vector<std::pair<int, long long> > >::iterator it =
        early.find(step);
    if (it != early.end()) {
      for (size_t k = 0; k < it->second.size(); k++)
        consumeNbr(it->second[k].first, step, it->second[k].second);
      early.erase(it);
    }
    maybeComplete();  // last action (may migrate)
  }

  void neighborVal(int fromIdx, int step, long long val) {
    // Never late (main advances only after everyone contributed); at most
    // one step early (sender needed broadcast step, which needed our
    // step-1 contribution).
    if (step != curStep && step != curStep + 1)
      CkPrintf("bcastred: elem %d unexpected step %d from %d at curStep %d\n",
               thisIndex, step, fromIdx, curStep);
    CkEnforce(step == curStep || step == curStep + 1);
    if (step == curStep + 1) {
      early[step].push_back(std::make_pair(fromIdx, val));
      return;
    }
    consumeNbr(fromIdx, step, val);
    maybeComplete();  // last action (may migrate)
  }

  void dumpState() {
    CkPrintf("bcastred DUMP: elem %d on PE %d: curStep %d bcastCount %d "
             "gotNbr %d contributed %d migrations %lld earlySteps %d\n",
             thisIndex, CkMyPe(), curStep, bcastCount, gotNbr,
             (int)contributed, migrations, (int)early.size());
  }

  void finish() {
    CkEnforce(curStep == nSteps - 1);
    CkEnforce(bcastCount == nSteps);
    CkEnforce(early.empty());
    long long expect = 0;
    for (int s = 0; s < nSteps; s++)
      expect += stepVal(thisIndex, s) + stepVal(left(), s) + stepVal(right(), s);
    if (checksum != expect)
      CkPrintf("bcastred: elem %d checksum MISMATCH got %lld expected %lld\n",
               thisIndex, checksum, expect);
    CkEnforce(checksum == expect);
    contribute(sizeof(long long), &migrations, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, finalStats), mainProxy));
  }
};

static void watchdogFire(void* arg, double t) { mainProxy.checkProgress(); }

#include "bcastred.def.h"
