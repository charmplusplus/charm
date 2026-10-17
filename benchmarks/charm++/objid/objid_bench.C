// objid_bench: per-operation costs of the chare-array paths that the 64-bit
// object id redesign (charm #3994, doc/objid64-design.md) touches, for a
// bounded (packed-id) or an unbounded (hashed-id) 1D array.
//
// Uses only public API, so it compiles against the pre-redesign runtime
// (per-PE id counter, home PE stored in the id) and the redesigned one.
//
// Usage: ./objid_bench +pe P [-k bounded|unbounded] [-n N] [-r R]
//   -k  bounded:   CkArrayOptions().setBounds(N): ids are the packed index
//                  (design section 2), home = map->homePe(idx).
//       unbounded: CkArrayOptions() with no bounds: ids are hashed
//                  (section 3); NEW mints from a per-process tranche and
//                  homes the element on rank 0 of the process chosen by the
//                  index hash; BASE mints from a per-PE counter and stores
//                  the creator PE in the id.
//   -n  elements (default 20000)    -r  timed repetitions (default 5)
//
// Each repetition creates a fresh array and runs the phases below once;
// repetition 0 is an untimed warm-up. Each phase prints the median over the
// R timed repetitions in microseconds per operation (CkWallTimer on PE 0),
// and a final line "objid_bench SUMMARY kind=... pes=... ..." collects them.
//
// Phases and the runtime path each one times:
//  1 insert  Every PE inserts its block of N/P indices locally
//            (proxy[i].insert(CkMyPe())), then main calls doneInserting and
//            waits for quiescence. us per insertion. Times id creation
//            (unbounded: tranche fetch_add on NEW, per-PE counter on BASE;
//            bounded: compress), record/cache insertion, and the informHome
//            message to the home PE.
//  2 cold    The Sender group on every PE sends one small message by index
//            to every element (N*P messages). A PE has never addressed the
//            remote elements: bounded computes the id locally and misses in
//            the location cache; unbounded has no id for the index on the
//            sender and buffers the message while it asks the home
//            (bufferForLocation -> requestLocation). Completion: a sum
//            reduction once every element has received P messages.
//            us per message.
//  3 warm    Phase 2 again: ids and locations are now cached on every
//            sender, so this is the direct by-id delivery path (section 4.3).
//  4 migrate Every element migrateMe's to (pe+1)%P (triggered by one array
//            broadcast); time to quiescence, including informHome at the
//            home. us per migration.
//  5 stale   Phase 2 after the migration: every sender's cache entry is one
//            hop stale, so each first message is forwarded by the old host
//            and the sender's cache is repaired (multiHop path).
//    stalewarm  A second send round right after stale, same senders and
//            elements: shows whether the stale round repaired the senders'
//            caches (direct delivery) or left them forwarding.
//    stale2  Two more migrations back to back (each to quiescence, untimed),
//            then phase 2 again: caches are two hops stale; NEW forwards to
//            the home after the first hop (section 4.3 multi-hop rule).
//    stale2warm  A second send round right after stale2 (as stalewarm).
//  6 bcast   100 array broadcasts of a small message, each followed by an
//            empty reduction back to main. us per broadcast. Control: the
//            broadcast path does not use per-element ids at the sender.
//
// A watchdog aborts if no phase completes for 120 s.
//
// Run (reconverse): ./objid_bench +pe 8 -k unbounded
//   multi-process:  lcrun -n 4 ./objid_bench +pe 8 -k unbounded

#include "objid_bench.decl.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Sender senderProxy;

static const int kBcasts = 100;
static const double kWatchdogSecs = 120.0;

static void watchdogFire(void*, double) { mainProxy.checkProgress(); }

class Elem : public CBase_Elem {
  int count = 0;  // messages received in the current send round
 public:
  Elem() {}
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void pup(PUP::er& p) { p | count; }

  void recv(int round) {
    if (++count == CkNumPes()) {
      int c = count;
      count = 0;
      contribute(sizeof(int), &c, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, roundDone), mainProxy));
    }
  }
  void doMigrate() { migrateMe((CkMyPe() + 1) % CkNumPes()); }
  void ping() { contribute(CkCallback(CkReductionTarget(Main, bcastDone), mainProxy)); }
};

class Sender : public CBase_Sender {
 public:
  Sender() {}
  static int lo(int n, int pe) { return (int)((long long)n * pe / CkNumPes()); }
  void insertAll(CProxy_Elem a, int n) {
    int me = CkMyPe();
    for (int i = lo(n, me); i < lo(n, me + 1); i++) a[i].insert(me);
    contribute(CkCallback(CkReductionTarget(Main, insertedAll), mainProxy));
  }
  void sendAll(CProxy_Elem a, int n, int round) {
    for (int i = 0; i < n; i++) a[i].recv(round);
  }
};

enum Phase { INSERT, COLD, WARM, MIGRATE, STALE, STALEWARM, MIG2A, MIG2B, STALE2, STALE2WARM,
             BCAST, NPHASE };
static const char* phaseName[NPHASE] = {"insert", "cold",  "warm",   "migrate",    "stale",
                                        "stalewarm", "mig2a", "mig2b", "stale2", "stale2warm",
                                        "bcast"};

class Main : public CBase_Main {
  std::string kind = "bounded";
  bool bounded = true;
  int N = 20000, R = 5;
  int rep = 0;  // 0 = warm-up
  int phase = INSERT;
  int round = 0, bcastsLeft = 0;
  double t0 = 0, lastProgress = 0;
  CProxy_Elem arr;
  std::vector<double> res[NPHASE];  // us per op, timed reps only

 public:
  Main(CkArgMsg* m) {
    char* k = nullptr;
    if (CmiGetArgString(m->argv, "-k", &k)) kind = k;
    CmiGetArgInt(m->argv, "-n", &N);
    CmiGetArgInt(m->argv, "-r", &R);
    delete m;
    if (kind != "bounded" && kind != "unbounded") CkAbort("objid_bench: -k bounded|unbounded");
    bounded = (kind == "bounded");
    CkEnforce(N >= CkNumPes() && R >= 1);
    mainProxy = thisProxy;
    senderProxy = CProxy_Sender::ckNew();
    CkPrintf("objid_bench: kind=%s pes=%d procs=%d n=%d reps=%d (+1 warm-up)\n", kind.c_str(),
             CkNumPes(), CkNumNodes(), N, R);
    lastProgress = CkWallTimer();
    CcdCallFnAfter(watchdogFire, NULL, 5000);
    startRep();
  }

  void checkProgress() {
    if (CkWallTimer() - lastProgress > kWatchdogSecs) {
      CkPrintf("objid_bench: no progress for %.0f s (rep %d phase %s round %d)\n", kWatchdogSecs,
               rep, phaseName[phase], round);
      fflush(stdout);
      CkAbort("objid_bench: hung");
    }
    CcdCallFnAfter(watchdogFire, NULL, 5000);
  }

  void record(double ops) {
    double now = CkWallTimer();
    if (rep > 0) res[phase].push_back((now - t0) * 1e6 / ops);
    lastProgress = now;
  }

  void startRep() {
    CkArrayOptions opts;
    if (bounded) opts.setBounds(N);
    arr = CProxy_Elem::ckNew(opts);
    phase = INSERT;
    t0 = CkWallTimer();
    senderProxy.insertAll(arr, N);
  }
  void insertedAll() {
    arr.doneInserting();
    CkStartQD(CkCallback(CkIndex_Main::insertQD(), thisProxy));
  }
  void insertQD() {
    record(N);
    startRound(COLD);
  }

  void startRound(int ph) {
    phase = ph;
    ++round;
    t0 = CkWallTimer();
    senderProxy.sendAll(arr, N, round);
  }
  void roundDone(int sum) {
    long long want = (long long)N * CkNumPes();
    if (sum != want) CkAbort("objid_bench: round %d got %d deliveries, want %lld", round, sum, want);
    record((double)want);
    if (phase == COLD) startRound(WARM);
    else if (phase == WARM) startMigrate(MIGRATE);
    else if (phase == STALE) startRound(STALEWARM);
    else if (phase == STALEWARM) startMigrate(MIG2A);
    else if (phase == STALE2) startRound(STALE2WARM);
    else startBcast();
  }

  void startMigrate(int ph) {
    phase = ph;
    t0 = CkWallTimer();
    arr.doMigrate();
    CkStartQD(CkCallback(CkIndex_Main::migrateQD(), thisProxy));
  }
  void migrateQD() {
    record(N);
    if (phase == MIGRATE) startRound(STALE);
    else if (phase == MIG2A) startMigrate(MIG2B);
    else startRound(STALE2);
  }

  void startBcast() {
    phase = BCAST;
    bcastsLeft = kBcasts;
    t0 = CkWallTimer();
    arr.ping();
  }
  void bcastDone() {
    if (--bcastsLeft > 0) { arr.ping(); return; }
    record(kBcasts);
    if (++rep <= R) startRep();
    else report();
  }

  static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  void report() {
    double med[NPHASE];
    for (int p = 0; p < NPHASE; p++) {
      med[p] = median(res[p]);
      if (p == MIG2A || p == MIG2B) continue;  // untimed setup for stale2
      auto mm = std::minmax_element(res[p].begin(), res[p].end());
      CkPrintf("objid_bench kind=%s phase=%-10s median %9.3f us/op  (min %.3f max %.3f, %d reps)\n",
               kind.c_str(), phaseName[p], med[p], *mm.first, *mm.second, R);
    }
    CkPrintf("objid_bench SUMMARY kind=%s pes=%d procs=%d n=%d insert=%.3f cold=%.3f warm=%.3f "
             "migrate=%.3f stale=%.3f stalewarm=%.3f stale2=%.3f stale2warm=%.3f bcast=%.3f\n",
             kind.c_str(), CkNumPes(), CkNumNodes(), N, med[INSERT], med[COLD], med[WARM],
             med[MIGRATE], med[STALE], med[STALEWARM], med[STALE2], med[STALE2WARM], med[BCAST]);
    CkExit();
  }
};

#include "objid_bench.def.h"
