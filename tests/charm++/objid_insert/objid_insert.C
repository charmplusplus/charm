// objid_insert: insertion, location lookup, stale-cache forwarding, demand
// creation and deletion of chare array elements under the 64-bit object id
// redesign (charm #3994, doc/objid64-design.md).
//
// Run twice: default = unbounded 2D array (hashed id kind, design section 3:
// ids minted from a per-process tranche, home = rank 0 of process
// indexHashKey(idx) % CkNumNodes(), section 3.3); -b = setBounds(1024,1024)
// (packed kind, section 2: the id is the packed index, home = map homePe).
//
// What is checked, and why:
//  Phase 1  Every PE inserts K elements (pe,k) locally and K elements
//           (pe+100,k) on the next PE. A local insert must be synchronous
//           (section 5, goal G5): ckLocal() is non-null right after insert.
//           Every element registers (index, element id, PE) with main, which
//           checks that all ids are distinct (section 3.1 uniqueness).
//  Phase 2  "Cold sends": every PE sends ping(round, fromPe) by index to every
//           element, including elements it never touched, so the source has
//           no id or location and must go through the home (section 4.1,
//           4.3; sendMsg -> handleUnknown -> bufferForLocation). Each element
//           enforces that each sender arrives exactly once per round; main
//           checks the ping total.
//  Phase 3  Elements migrate to pseudo-random other PEs, then phase 2 runs
//           over one-hop-stale caches; then two migrations back to back and
//           phase 2 again over two-hop-stale caches, which exercises the
//           "more than one hop -> forward to homePe(id)" rule in
//           CkArray::recvMsg (section 4.3). Five cycles, different seeds.
//  Phase 4  Demand creation: a [createhome] entry sent from every PE to
//           indices (300+pe,k) that were never inserted; each must be created
//           exactly once, at its home PE (section 5, "Demand creation").
//           Then phase 2 over the enlarged set.
//  Phase 5  Half the elements ckDestroy() themselves; phase 2 over the
//           survivors (section 4.2 reclaimRemote, section 5 "Deletion").
//
// Options: -b packed kind; -k K elements per PE per pattern (default 16);
//          -w watchdog stall seconds (default 60, 0 disables);
//          -D skip phase 4.
//
// Phase 4 in the hashed kind exposed a defect that predates the redesign: a
// non-home source with no id for a never-created index buffers the message
// (bufferForCreation) and sends requestDemandCreation(idx, ctor, pe = home) to
// the home, which creates the element but, for createhome, told nobody; the
// requester's bufferedCreationMsgs entry was never flushed, so only the home
// PE's own revive arrived. The packed kind never hit it because the source
// has the id and forwards the small message to the home. Fixed alongside this
// test: bufferForCreation now also requests the location by index, which the
// home answers once the element exists.
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <stdio.h>
#include "pup_stl.h"
#include "objid_insert.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Elem arrProxy;
/* readonly */ CProxy_Inserter inserterProxy;
/* readonly */ int K;
/* readonly */ int bounded;

static unsigned long long mix64(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static unsigned long long mixIdx(int x, int y, int seed) {
  return mix64(((unsigned long long)(unsigned)x << 32) ^ (unsigned)y ^
               mix64((unsigned long long)(unsigned)seed));
}
// Destination of a migration: never the current PE.
static int migDest(int x, int y, int seed, int cur) {
  int P = CkNumPes();
  return (cur + 1 + (int)(mixIdx(x, y, seed) % (unsigned long long)(P - 1))) % P;
}
static bool diesAt(int x, int y, int seed) { return (mixIdx(x, y, seed) >> 7) & 1; }

static void watchdogFire(void* arg, double t);
static void abortFire(void* arg, double t) {
  fflush(stdout);
  CkAbort("objid_insert: hung (state dumped above)");
}

enum Action { PING, MIGRATE, REVIVE, REPORT, DESTROY, DONE };
struct Step { Action a; int seed; const char* what; };

class Main : public CBase_Main {
  struct Reg { CmiUInt8 id; int pe; int demand; };
  std::map<std::pair<int, int>, Reg> regs;
  std::set<CmiUInt8> ids;
  std::set<std::pair<int, int>> live;
  std::vector<Step> script;
  size_t cur = 0;
  int round = 0;
  int nMigrateRounds = 0;
  bool doDemand = true;
  bool destroyed = false;
  int wdSecs = 60;
  double lastAdvance = 0;
  bool dumped = false;
  int P;

public:
  Main(CkArgMsg* m) {
    setvbuf(stdout, NULL, _IONBF, 0);
    K = 16;
    bounded = CmiGetArgFlag(m->argv, "-b") ? 1 : 0;
    CmiGetArgInt(m->argv, "-k", &K);
    CmiGetArgInt(m->argv, "-w", &wdSecs);
    doDemand = !CmiGetArgFlag(m->argv, "-D");
    delete m;
    P = CkNumPes();
    CkEnforce(K >= 1 && K <= 1024);
    CkEnforce(P <= 700);  // indices 300+pe must stay inside 1024 bounds
    CkPrintf("objid_insert: %d PEs, %d processes, K=%d, %s kind%s\n", P,
             CkNumNodes(), K, bounded ? "packed (setBounds 1024x1024)" : "hashed (unbounded)",
             doDemand ? "" : ", phase 4 (demand creation) skipped (-D)");
    mainProxy = thisProxy;
    CkArrayOptions opts;
    if (bounded) opts.setBounds(1024, 1024);
    arrProxy = CProxy_Elem::ckNew(opts);
    inserterProxy = CProxy_Inserter::ckNew();

    script.push_back({PING, 0, "cold sends"});
    if (P > 1) {
      for (int c = 0; c < 5; c++) {
        int s = 1000 + 10 * c;
        script.push_back({MIGRATE, s, "migrate"});
        script.push_back({PING, 0, "one-hop-stale sends"});
        script.push_back({MIGRATE, s + 1, "migrate"});
        script.push_back({MIGRATE, s + 2, "migrate"});
        script.push_back({PING, 0, "two-hop-stale sends"});
      }
    } else {
      CkPrintf("objid_insert: 1 PE, migration phases skipped\n");
    }
    if (doDemand) {
      script.push_back({REVIVE, 0, "demand creation"});
      script.push_back({REPORT, 0, "demand-creation report"});
      script.push_back({PING, 0, "sends after demand creation"});
    }
    script.push_back({DESTROY, 77, "destroy half"});
    script.push_back({PING, 0, "sends to survivors"});
    script.push_back({REPORT, 0, "final report"});
    script.push_back({DONE, 0, "done"});

    lastAdvance = CkWallTimer();
    if (wdSecs > 0) CcdCallFnAfter(watchdogFire, NULL, 2000);
    inserterProxy.phase1();
  }

  void progress() { lastAdvance = CkWallTimer(); }

  void registerElem(int x, int y, CmiUInt8 id, int pe, int demand) {
    progress();
    auto key = std::make_pair(x, y);
    if (regs.count(key)) {
      CkPrintf("objid_insert FAIL: element (%d,%d) registered twice (ids %llx on PE %d, "
               "%llx on PE %d)\n", x, y, (unsigned long long)regs[key].id, regs[key].pe,
               (unsigned long long)id, pe);
      CkAbort("duplicate element");
    }
    if (!ids.insert(id).second) {
      CkPrintf("objid_insert FAIL: element id %llx of (%d,%d) is not unique\n",
               (unsigned long long)id, x, y);
      CkAbort("duplicate id");
    }
    regs[key] = Reg{id, pe, demand};
    live.insert(key);
  }

  void phase1Done() {
    progress();
    arrProxy.doneInserting();
    CkStartQD(CkCallback(CkIndex_Main::quiescent(), thisProxy));
    cur = (size_t)-1;  // quiescent() advances to script[0]
  }

  void quiescent() {
    progress();
    if (cur == (size_t)-1) {
      CkEnforce(regs.size() == (size_t)(2 * K * P));
      for (int pe = 0; pe < P; pe++)
        for (int k = 0; k < K; k++) {
          auto a = regs.find(std::make_pair(pe, k));
          auto b = regs.find(std::make_pair(pe + 100, k));
          CkEnforce(a != regs.end() && b != regs.end());
          CkEnforce(a->second.pe == pe);
          CkEnforce(b->second.pe == (pe + 1) % P);
        }
      CkPrintf("objid_insert: phase 1 ok, %zu elements, ids distinct\n", regs.size());
      cur = 0;
      runStep();
      return;
    }
    Step& s = script[cur];
    if (s.a == REVIVE) {
      int nd = 0;
      for (auto& r : regs) nd += r.second.demand;
      CkEnforce(nd == P * K);
      for (int pe = 0; pe < P; pe++)
        for (int k = 0; k < K; k++) {
          auto it = regs.find(std::make_pair(300 + pe, k));
          CkEnforce(it != regs.end() && it->second.demand == 1);
        }
    }
    next();
  }

  void next() { cur++; runStep(); }

  void runStep() {
    progress();
    Step& s = script[cur];
    switch (s.a) {
      case PING: {
        round++;
        std::vector<int> xs, ys;
        for (auto& e : live) { xs.push_back(e.first); ys.push_back(e.second); }
        CkPrintf("objid_insert: round %d (%s) over %zu elements\n", round, s.what,
                 live.size());
        inserterProxy.sendPings(round, xs, ys);
        break;
      }
      case MIGRATE:
        nMigrateRounds++;
        arrProxy.migrateNow(s.seed);
        CkStartQD(CkCallback(CkIndex_Main::quiescent(), thisProxy));
        break;
      case REVIVE:
        CkPrintf("objid_insert: demand-creating %d elements from every PE\n", P * K);
        inserterProxy.sendRevives();
        CkStartQD(CkCallback(CkIndex_Main::quiescent(), thisProxy));
        break;
      case REPORT:
        arrProxy.report();
        break;
      case DESTROY: {
        size_t before = live.size();
        for (auto it = live.begin(); it != live.end();)
          if (diesAt(it->first, it->second, s.seed)) it = live.erase(it);
          else ++it;
        CkEnforce(!live.empty() && live.size() < before);
        destroyed = true;
        CkPrintf("objid_insert: destroying %zu of %zu elements\n", before - live.size(),
                 before);
        arrProxy.maybeDie(s.seed);
        CkStartQD(CkCallback(CkIndex_Main::quiescent(), thisProxy));
        break;
      }
      case DONE:
        CkPrintf("objid_insert PASS (%s kind, %d PEs, %d processes, %d ping rounds, "
                 "%d migration rounds)\n", bounded ? "packed" : "hashed", P, CkNumNodes(),
                 round, nMigrateRounds);
        CkExit();
        break;
    }
  }

  void pingDone(int nElems, int nPings) {
    progress();
    if (nElems != (int)live.size() || nPings != (int)live.size() * P) {
      CkPrintf("objid_insert FAIL: round %d: %d elements reported %d pings, expected "
               "%zu elements, %zu pings\n", round, nElems, nPings, live.size(),
               live.size() * P);
      CkAbort("ping count mismatch");
    }
    next();
  }

  void reportDone(int nElems, int nRevives, int nDemand, int nMigrations) {
    progress();
    CkPrintf("objid_insert: report: %d elements, %d revives, %d demand-created, "
             "%d migrations\n", nElems, nRevives, nDemand, nMigrations);
    CkEnforce(nElems == (int)live.size());
    if (!destroyed && doDemand) {
      CkEnforce(nDemand == P * K);
      CkEnforce(nRevives == P * P * K);
    }
    next();
  }

  void checkProgress() {
    if (!dumped && wdSecs > 0 && CkWallTimer() - lastAdvance > (double)wdSecs) {
      dumped = true;
      CkPrintf("objid_insert WATCHDOG: no progress for %d s; script step %zu (%s), "
               "round %d, %zu registered, %zu live. Dumps:\n", wdSecs, cur,
               cur < script.size() ? script[cur].what : "phase 1", round, regs.size(),
               live.size());
      inserterProxy.dumpState();
      arrProxy.dumpState();
      CcdCallFnAfter(abortFire, NULL, 3000);
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }
};

class Inserter : public CBase_Inserter {
  int lastRound = 0;
public:
  Inserter() {}

  void phase1() {
    int me = CkMyPe(), P = CkNumPes();
    for (int k = 0; k < K; k++) {
      arrProxy(me, k).insert(0, me);
      // Local insertion is synchronous (design section 5, goal G5).
      CkEnforce(arrProxy(me, k).ckLocal() != nullptr);
    }
    for (int k = 0; k < K; k++) arrProxy(me + 100, k).insert(0, (me + 1) % P);
    contribute(CkCallback(CkReductionTarget(Main, phase1Done), mainProxy));
  }

  void sendPings(int round, std::vector<int> xs, std::vector<int> ys) {
    lastRound = round;
    for (size_t i = 0; i < xs.size(); i++) arrProxy(xs[i], ys[i]).ping(round, CkMyPe());
  }

  void sendRevives() {
    for (int pe = 0; pe < CkNumPes(); pe++)
      for (int k = 0; k < K; k++) arrProxy(300 + pe, k).revive(CkMyPe());
  }

  void dumpState() {
    CkPrintf("objid_insert DUMP: Inserter PE %d lastRound %d local elements %u\n",
             CkMyPe(), lastRound, arrProxy.numLocalElements());
  }
};

class Elem : public CBase_Elem {
  std::map<int, std::vector<char>> pending;  // round -> senders seen
  std::vector<char> reviveFrom;
  int revives = 0;
  int demand = 0;
  int migrations = 0;
  int lastRound = 0;

  void registerMe() {
    mainProxy.registerElem(thisIndex.x, thisIndex.y, ckGetID().getElementID(), CkMyPe(),
                           demand);
  }

public:
  Elem(int origin) : reviveFrom(CkNumPes(), 0) {
    CkEnforce(thisIndex.x < 300);
    registerMe();
  }
  // Demand creation ([createhome] revive): must run at the element's home PE.
  Elem() : reviveFrom(CkNumPes(), 0) {
    demand = 1;
    CkEnforce(thisIndex.x >= 300);
    int home = arrProxy.ckLocMgr()->homePe(thisIndexMax);
    if (CkMyPe() != home)
      CkPrintf("objid_insert FAIL: (%d,%d) demand-created on PE %d, home is PE %d\n",
               thisIndex.x, thisIndex.y, CkMyPe(), home);
    CkEnforce(CkMyPe() == home);
    registerMe();
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p) {
    p | pending;
    p | reviveFrom;
    p | revives;
    p | demand;
    p | migrations;
    p | lastRound;
  }

  void ping(int round, int fromPe) {
    CkEnforce(fromPe >= 0 && fromPe < CkNumPes());
    std::vector<char>& seen = pending[round];
    if (seen.empty()) seen.assign(CkNumPes(), 0);
    if (seen[fromPe]) {
      CkPrintf("objid_insert FAIL: (%d,%d) got round %d ping from PE %d twice\n",
               thisIndex.x, thisIndex.y, round, fromPe);
      CkAbort("duplicate ping");
    }
    seen[fromPe] = 1;
    int n = 0;
    for (char c : seen) n += c;
    if (n == CkNumPes()) {
      pending.erase(round);
      lastRound = round;
      int v[2] = {1, n};
      contribute(sizeof(v), v, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, pingDone), mainProxy));
    }
  }

  void revive(int fromPe) {
    CkEnforce(demand == 1);
    CkEnforce(fromPe >= 0 && fromPe < CkNumPes());
    CkEnforce(!reviveFrom[fromPe]);
    reviveFrom[fromPe] = 1;
    revives++;
  }

  void migrateNow(int seed) {
    if (CkNumPes() < 2) return;
    migrations++;
    migrateMe(migDest(thisIndex.x, thisIndex.y, seed, CkMyPe()));  // last action
  }

  void maybeDie(int seed) {
    // The proxy form (a message to itself, as in the manual and the other tests).
    // A direct ckDestroy() here crashes in broadcast delivery (null CkLocRec in
    // CkLocRec::invokeEntry, called from CkArrayBroadcaster::attemptDelivery),
    // on the pre-redesign runtime as well.
    if (diesAt(thisIndex.x, thisIndex.y, seed)) thisProxy[thisIndex].ckDestroy();
  }

  void report() {
    int v[4] = {1, revives, demand, migrations};
    contribute(sizeof(v), v, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, reportDone), mainProxy));
  }

  void dumpState() {
    CkPrintf("objid_insert DUMP: (%d,%d) id %llx on PE %d lastRound %d pending rounds "
             "%zu migrations %d demand %d revives %d\n", thisIndex.x, thisIndex.y,
             (unsigned long long)ckGetID().getElementID(), CkMyPe(), lastRound,
             pending.size(), migrations, demand, revives);
    for (auto& r : pending) {
      int n = 0;
      for (char c : r.second) n += c;
      CkPrintf("objid_insert DUMP:   (%d,%d) round %d has %d/%d pings\n", thisIndex.x,
               thisIndex.y, r.first, n, CkNumPes());
    }
  }
};

static void watchdogFire(void* arg, double t) { mainProxy.checkProgress(); }

#include "objid_insert.def.h"
