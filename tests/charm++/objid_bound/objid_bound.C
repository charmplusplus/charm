// objid_bound: bound arrays over a hashed-kind (unbounded) array, under
// manual migration and under load-balancer migration.
//
// What is exercised (doc/objid64-design.md, 64-bit object id redesign):
//
//   - Array A is 1D with NO bounds in its CkArrayOptions, so it is the
//     "hashed kind" (section 3): ids are minted from a per-process tranche,
//     carry no PE number, and an element's home is rank 0 of process
//     hashKey(idx) % CkNumNodes().
//   - Array B is created with CkArrayOptions().bindTo(A), so A and B share
//     one location manager. When an A element migrates, its B sibling
//     migrates with it (section 5, Migration). A sibling that does not yet
//     exist on the destination is demand-created there by ckarray.C
//     deliverInline's bound-sibling path, which recovers the index from the
//     id through the location manager's local record -- the one remaining
//     use of the reverse lookup lookupIdx (section 3.4). If that lookup has
//     no record, it aborts with "index of id ... is not known on PE ...".
//   - Two migration entry paths: default mode uses migrateMe() from user
//     code; -lb mode uses AtSync() and +balancer (RotateLB moves every
//     object each step, GreedyLB moves some). User code never calls
//     migrateMe while a +balancer is active.
//
// Per round r (30 rounds): main broadcasts step(r) to A. A[i] sends
// fromA(r, i, CkMyPe()) to B[i] by index; B[i] enforces the sender PE equals
// its own PE (bound siblings are always co-located) and that its pupped
// round counter is continuous (a B element demand-created fresh instead of
// migrated would lose that state), then sends fromB to A[(i+1)%n]. A[j],
// once it has both step(r) and its fromB(r), checks the payload and
// contributes i to a sum reduction; main checks the closed form.
//   default: main then broadcasts migrateNow(r) to A only; each A element
//            migrates to a pseudo-random other PE (migrateMe is its last
//            action); B never calls migrateMe. Main waits for quiescence
//            and starts the next round.
//   -lb:     each A element calls AtSync() right after contributing; its
//            ResumeFromSync() contributes to a "resumed" reduction, which
//            starts the next round. B has usesAtSync = false.
// Array C, also bound to A, is NEVER inserted. Every round, A[i] (after
// handling step(r)) sends C[i].poke(r, CkMyPe()), an entry marked
// [createhome]. The first poke arrives on the PE of A[i], where the shared
// location manager knows the id's location (the sibling's) but C has no
// element: ckarray.C deliverInline's bound-sibling branch recovers the index
// with locMgr->lookupIdx(id) (the local-record branch, section 3.4) and
// demand-creates C[i] with its default constructor. C enforces that the
// sender PE equals its own PE and that it gets exactly one poke per round,
// and acks A[i], which contributes only after the ack -- so C[i] exists
// before any migration, and every later A migration must carry it.
// At the end A and B report their migration counts (counted in pup when
// unpacking a migrated element). Enforced: A's count > 0, and B's count ==
// A's count == C's count (every A migration carried its siblings), and
// exactly nElems C elements were created, each exactly once. C reports by
// point-to-point messages to main rather than reductions, because a
// reduction over a never-inserted, demand-created array has no settled
// membership to count against.
//
// Flags: -lb (load-balancer mode), -noc (leave array C out; runs the A/B
//        checks alone), -n <elements, default 24>,
//        -r <rounds, default 30>, -w <watchdog seconds, default 60>
//
// Run (reconverse): ./objid_bound +pe 4 [-lb +balancer RotateLB]
//   multi-process   lcrun -n 2 ./objid_bound +pe 4 [-lb +balancer RotateLB]

#include "objid_bound.decl.h"
#include <stdio.h>
#include <vector>

// Every process: unbuffered stdout, so a multi-process abort does not discard
// the other processes' last lines (they are block-buffered under lcrun).
void unbufferStdout(void) { setvbuf(stdout, NULL, _IONBF, 0); }

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_A aProxy;
/*readonly*/ CProxy_B bProxy;
/*readonly*/ CProxy_C cProxy;
/*readonly*/ int nElems;
/*readonly*/ int nRounds;
/*readonly*/ int lbMode;
/*readonly*/ int useC;

static inline unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Payload of a message originating at index i in round r.
static inline long long payload(int i, int r) {
  return (long long)(mix64(((unsigned long long)(unsigned)r << 32) |
                           (unsigned)i) % 1000003ULL);
}

static void watchdogFire(void*, double) { mainProxy.checkProgress(); }
static void abortFire(void*, double) {
  fflush(stdout);
  CkAbort("objid_bound: hung (element dumps above)");
}

class Main : public CBase_Main {
  int round = 0;
  bool sumSeen = false;   // roundSum for the current round arrived
  double t0 = 0, lastAdvance = 0;
  int wdSecs = 60;
  bool dumped = false;
  long long migA = -1, migB = -1;
  int cCreatedCount = 0, cStatsCount = 0;
  long long migC = 0;
  std::vector<int> cSeen;

 public:
  Main(CkArgMsg* m) {
    setvbuf(stdout, NULL, _IONBF, 0);
    nElems = 24;
    nRounds = 30;
    lbMode = CmiGetArgFlag(m->argv, "-lb") ? 1 : 0;
    useC = CmiGetArgFlag(m->argv, "-noc") ? 0 : 1;
    CmiGetArgInt(m->argv, "-n", &nElems);
    CmiGetArgInt(m->argv, "-r", &nRounds);
    CmiGetArgInt(m->argv, "-w", &wdSecs);
    delete m;
    CkEnforce(nElems >= 2 && nRounds >= 1);
    CkEnforce(CkNumPes() >= 2);  // the test is about migration
    CkPrintf("objid_bound: %d PEs, %d processes, %d elements, %d rounds, "
             "mode %s%s\n", CkNumPes(), CkNumNodes(), nElems, nRounds,
             lbMode ? "load balancer (AtSync)" : "manual (migrateMe)",
             useC ? ", with demand-created bound array C" : ", no array C");
    mainProxy = thisProxy;

    // A: no bounds -> hashed kind. Insert round-robin, then close insertion.
    aProxy = CProxy_A::ckNew(CkArrayOptions());
    for (int i = 0; i < nElems; i++) aProxy[i].insert(i % CkNumPes());
    aProxy.doneInserting();

    // B: bound to A (shared location manager). Insert at the same PE as the
    // A sibling.
    CkArrayOptions bopts;
    bopts.bindTo(aProxy);
    bProxy = CProxy_B::ckNew(bopts);
    for (int i = 0; i < nElems; i++) bProxy[i].insert(i % CkNumPes());
    bProxy.doneInserting();

    // C: bound to A, never inserted, no doneInserting. Demand-created only.
    CkArrayOptions copts;
    copts.bindTo(aProxy);
    cProxy = CProxy_C::ckNew(copts);
    cSeen.assign(nElems, 0);

    t0 = lastAdvance = CkWallTimer();
    if (wdSecs > 0) CcdCallFnAfter(watchdogFire, NULL, 2000);
    aProxy.step(0);
  }

  void checkProgress() {
    if (!dumped && CkWallTimer() - lastAdvance > (double)wdSecs) {
      dumped = true;
      CkPrintf("objid_bound WATCHDOG: no progress for %d s; main at round %d "
               "(sumSeen %d). Element dumps:\n", wdSecs, round, (int)sumSeen);
      aProxy.dumpState();
      bProxy.dumpState();
      cProxy.dumpState();
      CkPrintf("objid_bound WATCHDOG: %d of %d C elements created, %d C "
               "reports\n", cCreatedCount, nElems, cStatsCount);
      CcdCallFnAfter(abortFire, NULL, 3000);
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void roundSum(long long sum) {
    long long expect = (long long)nElems * (nElems - 1) / 2;
    if (sum != expect)
      CkPrintf("objid_bound: round %d sum MISMATCH got %lld expected %lld\n",
               round, sum, expect);
    CkEnforce(sum == expect);
    CkEnforce(!sumSeen);
    sumSeen = true;
    lastAdvance = CkWallTimer();
    if (lbMode) return;  // next round starts from resumed()
    aProxy.migrateNow(round);
    CkStartQD(CkCallback(CkIndex_Main::qdDone(), thisProxy));
  }

  void qdDone() {
    CkEnforce(!lbMode && sumSeen);
    nextRound();
  }

  void resumed(long long roundSum) {
    CkEnforce(lbMode && sumSeen);
    long long expect = (long long)nElems * round;
    if (roundSum != expect)
      CkPrintf("objid_bound: resumed MISMATCH at round %d: sum of element "
               "rounds %lld expected %lld\n", round, roundSum, expect);
    CkEnforce(roundSum == expect);
    nextRound();
  }

  void nextRound() {
    lastAdvance = CkWallTimer();
    sumSeen = false;
    round++;
    if (round < nRounds) aProxy.step(round);
    else {
      aProxy.report();
      bProxy.report();
      if (useC) for (int i = 0; i < nElems; i++) cProxy[i].report();
    }
  }

  void statsA(long long m) { migA = m; maybeFinish(); }
  void statsB(long long m) { migB = m; maybeFinish(); }

  void cCreated(int idx, int pe) {
    CkEnforce(idx >= 0 && idx < nElems);
    if (cSeen[idx] != 0)
      CkPrintf("objid_bound: C[%d] created twice (second on PE %d)\n", idx, pe);
    CkEnforce(cSeen[idx] == 0);
    cSeen[idx] = 1;
    cCreatedCount++;
    CkEnforce(round == 0);  // the first poke happens in round 0
  }

  void cStats(int idx, int pokes, long long m) {
    CkEnforce(idx >= 0 && idx < nElems && cSeen[idx] == 1);
    CkEnforce(pokes == nRounds);
    cSeen[idx] = 2;
    migC += m;
    cStatsCount++;
    maybeFinish();
  }

  void maybeFinish() {
    if (migA < 0 || migB < 0 || (useC && cStatsCount < nElems)) return;
    if (!useC) migC = migA, cCreatedCount = nElems;  // nothing to compare
    if (cCreatedCount != nElems || migC != migA)
      CkPrintf("objid_bound: C MISMATCH created %d (expected %d), migrations "
               "%lld (A %lld)\n", cCreatedCount, nElems, migC, migA);
    CkEnforce(cCreatedCount == nElems);
    CkEnforce(migC == migA);
    if (migA != migB)
      CkPrintf("objid_bound: migration count MISMATCH A %lld B %lld\n",
               migA, migB);
    CkEnforce(migA > 0);
    CkEnforce(migA == migB);
    if (useC)
      CkPrintf("objid_bound PASS: %s, %d rounds, %lld migrations (A) = %lld "
               "(B) = %lld (C), %d C demand-created, %.3f s\n",
               lbMode ? "lb" : "manual", nRounds, migA, migB, migC,
               cCreatedCount, CkWallTimer() - t0);
    else
      CkPrintf("objid_bound PASS: %s, %d rounds, %lld migrations (A) = %lld "
               "(B), no C, %.3f s\n", lbMode ? "lb" : "manual", nRounds,
               migA, migB, CkWallTimer() - t0);
    CkExit();
  }
};

class A : public CBase_A {
  int stepRound = -1;    // last step(r) received
  int fromBRound = -1;   // last fromB(r) received
  int doneRound = -1;    // last round contributed
  int ackRound = -1;     // last pokeAck(r) received from C[i]
  long long migrations = 0;

  int left() const { return (thisIndex + nElems - 1) % nElems; }

  void maybeContribute() {
    if (stepRound != fromBRound || stepRound != ackRound ||
        doneRound == stepRound) return;
    doneRound = stepRound;
    long long v = thisIndex;
    contribute(sizeof(v), &v, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, roundSum), mainProxy));
    if (lbMode) AtSync();  // last action
  }

 public:
  A() { usesAtSync = lbMode ? true : false; }
  A(CkMigrateMessage* m) : CBase_A(m) {}

  void pup(PUP::er& p) {
    p | stepRound; p | fromBRound; p | doneRound; p | ackRound; p | migrations;
    if (p.isUnpacking()) migrations++;
  }

  void step(int r) {
    if (r != stepRound + 1)
      CkPrintf("objid_bound: A[%d] step misorder got %d after %d\n",
               thisIndex, r, stepRound);
    CkEnforce(r == stepRound + 1);
    stepRound = r;
    bProxy[thisIndex].fromA(r, thisIndex, CkMyPe(), payload(thisIndex, r));
    if (useC) cProxy[thisIndex].poke(r, CkMyPe());
    else ackRound = r;
    maybeContribute();
  }

  void fromB(int r, int srcIdx, int srcPe, long long val) {
    if (srcIdx != left() || r != fromBRound + 1 || val != payload(srcIdx, r))
      CkPrintf("objid_bound: A[%d] bad fromB r %d (expected %d) src %d "
               "(expected %d) val %lld (expected %lld)\n", thisIndex, r,
               fromBRound + 1, srcIdx, left(), val, payload(srcIdx, r));
    CkEnforce(srcIdx == left());
    CkEnforce(r == fromBRound + 1);
    CkEnforce(val == payload(srcIdx, r));
    CkEnforce(srcPe >= 0 && srcPe < CkNumPes());
    fromBRound = r;
    maybeContribute();
  }

  void pokeAck(int r, int cIdx, int cPe) {
    if (cIdx != thisIndex || cPe != CkMyPe() || r != ackRound + 1)
      CkPrintf("objid_bound: A[%d] on PE %d bad pokeAck r %d (expected %d) "
               "from C[%d] on PE %d\n", thisIndex, CkMyPe(), r, ackRound + 1,
               cIdx, cPe);
    CkEnforce(cIdx == thisIndex);
    CkEnforce(cPe == CkMyPe());
    CkEnforce(r == ackRound + 1);
    ackRound = r;
    maybeContribute();
  }

  void migrateNow(int r) {
    CkEnforce(!lbMode && r == doneRound);
    int hop = 1 + (int)(mix64(((unsigned long long)(unsigned)r << 32) ^
                              (unsigned)thisIndex) % (CkNumPes() - 1));
    int dest = (CkMyPe() + hop) % CkNumPes();
    migrateMe(dest);  // last action
  }

  void ResumeFromSync() {
    CkEnforce(lbMode);
    long long v = doneRound;  // main checks the sum == nElems * round
    contribute(sizeof(v), &v, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, resumed), mainProxy));
  }

  void report() {
    CkEnforce(doneRound == nRounds - 1);
    contribute(sizeof(migrations), &migrations, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, statsA), mainProxy));
  }

  void dumpState() {
    CkPrintf("objid_bound DUMP: A[%d] on PE %d: step %d fromB %d ack %d "
             "done %d migrations %lld\n", thisIndex, CkMyPe(), stepRound,
             fromBRound, ackRound, doneRound, migrations);
  }
};

class B : public CBase_B {
  int lastRound = -1;
  long long migrations = 0;

 public:
  B() { usesAtSync = false; }
  B(CkMigrateMessage* m) : CBase_B(m) {}

  void pup(PUP::er& p) {
    p | lastRound; p | migrations;
    if (p.isUnpacking()) migrations++;
  }

  void fromA(int r, int srcIdx, int srcPe, long long val) {
    if (srcPe != CkMyPe() || srcIdx != thisIndex || r != lastRound + 1)
      CkPrintf("objid_bound: B[%d] on PE %d got fromA r %d (expected %d) "
               "from A[%d] on PE %d -- bound siblings not co-located or state "
               "lost\n", thisIndex, CkMyPe(), r, lastRound + 1, srcIdx, srcPe);
    CkEnforce(srcIdx == thisIndex);
    CkEnforce(srcPe == CkMyPe());
    CkEnforce(r == lastRound + 1);
    CkEnforce(val == payload(srcIdx, r));
    lastRound = r;
    int next = (thisIndex + 1) % nElems;
    aProxy[next].fromB(r, thisIndex, CkMyPe(), payload(thisIndex, r));
  }

  void report() {
    CkEnforce(lastRound == nRounds - 1);
    contribute(sizeof(migrations), &migrations, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, statsB), mainProxy));
  }

  void dumpState() {
    CkPrintf("objid_bound DUMP: B[%d] on PE %d: lastRound %d migrations "
             "%lld\n", thisIndex, CkMyPe(), lastRound, migrations);
  }
};

class C : public CBase_C {
  int lastRound = -1;
  int pokes = 0;
  long long migrations = 0;

 public:
  C() {
    usesAtSync = false;
    CkPrintf("objid_bound: C[%d] demand-created on PE %d\n", thisIndex,
             CkMyPe());
    mainProxy.cCreated(thisIndex, CkMyPe());
  }
  C(CkMigrateMessage* m) : CBase_C(m) {}

  void pup(PUP::er& p) {
    p | lastRound; p | pokes; p | migrations;
    if (p.isUnpacking()) migrations++;
  }

  void poke(int r, int senderPe) {
    if (senderPe != CkMyPe() || r != lastRound + 1)
      CkPrintf("objid_bound: C[%d] on PE %d got poke r %d (expected %d) from "
               "PE %d\n", thisIndex, CkMyPe(), r, lastRound + 1, senderPe);
    CkEnforce(senderPe == CkMyPe());
    CkEnforce(r == lastRound + 1);  // exactly one poke per round, in order
    lastRound = r;
    pokes++;
    aProxy[thisIndex].pokeAck(r, thisIndex, CkMyPe());
  }

  void report() {
    CkEnforce(lastRound == nRounds - 1);
    mainProxy.cStats(thisIndex, pokes, migrations);
  }

  void dumpState() {
    CkPrintf("objid_bound DUMP: C[%d] on PE %d: lastRound %d pokes %d "
             "migrations %lld\n", thisIndex, CkMyPe(), lastRound, pokes,
             migrations);
  }
};

#include "objid_bound.def.h"
