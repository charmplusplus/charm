// Broadcasts with priorities, and quiescence after broadcasts.
//
// Part 1, priority order. Every PE runs G::hold, which keeps the PE busy for
// a while. The sending PE issues K broadcasts from inside its hold, the least
// urgent first, so that they are all queued on it by the time hold returns;
// they must then run most urgent first. Kinds: a group broadcast with
// marshalled parameters (a nokeep message, which the PEs of a process share),
// a group broadcast of a message (each PE gets a copy), a nodegroup
// broadcast, and group broadcasts with LIFO queueing and no priority, which
// must run in the reverse of the order they were issued in. After a wrong
// order the round is repeated with a longer hold, and the test fails only if
// the longest hold still shows one.
// The order is checked only in the sender's process, and only when that
// process has one PE: messages from another process arrive in no defined
// order, and in a process with several PEs one PE hands the broadcasts to the
// others one at a time, so a PE can run the first before the second arrives.
// Every PE still runs all K broadcasts and replies after the last.
// Bit-vector priorities are not tested: reconverse queues them at priority 0.
//
// Part 2, quiescence. One PE issues a burst of group, nodegroup and array
// broadcasts, some with priorities. Quiescence must be detected, and only
// after every copy has run.

#include <atomic>
#include <mutex>
#include <vector>
#include "bcast_priority.decl.h"

CProxy_Main mainProxy;
CProxy_G gProxy;
CProxy_NG ngProxy;
CProxy_A aProxy;

static const int K = 8;  // broadcasts per round
static const int NKINDS = 4;
static const int MAXTRIES = 6;
static const int BURST = 200;
static const char *kindName[NKINDS] = {"group, marshalled", "group, message", "nodegroup",
                                       "group, LIFO"};

CkpvDeclare(long, arrayCount);

struct PMsg : public CMessage_PMsg {
  int p, sender;
};

static bool inOrder(const std::vector<int> &seq) {
  for (size_t i = 1; i < seq.size(); i++)
    if (seq[i - 1] > seq[i]) return false;
  return true;
}

class Main : public CBase_Main {
  int round = 0, tries = 0, replies = 0, expected = 0;
  bool allOk = true;
  double secs = 0.02;
  long total[3] = {0, 0, 0};

 public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;
    gProxy = CProxy_G::ckNew();
    ngProxy = CProxy_NG::ckNew();
    aProxy = CProxy_A::ckNew(2 * CkNumPes());
    CkStartQD(CkCallback(CkIndex_Main::startRound(), thisProxy));
  }

  // rounds: each kind, sent from PE 0 and from the last PE
  void startRound() {
    if (round == 2 * NKINDS) {
      gProxy[CkNumPes() - 1].burst(BURST);
      CkStartQD(CkCallback(CkIndex_Main::quiet(), thisProxy));
      return;
    }
    int kind = round / 2;
    int sender = (round % 2) ? CkNumPes() - 1 : 0;
    replies = 0;
    allOk = true;
    expected = (kind == 2) ? CkNumNodes() : CkNumPes();
    gProxy.hold(kind, sender, secs);
  }

  void roundReply(bool ok) {
    allOk = allOk && ok;
    if (++replies < expected) return;
    if (allOk) {
      round++;
      tries = 0;
      secs = 0.02;
    } else {
      if (++tries == MAXTRIES)
        CkAbort("bcast_priority: %s broadcasts did not run in priority order",
                kindName[round / 2]);
      secs *= 3;
    }
    thisProxy.startRound();
  }

  void quiet() {
    replies = 0;
    gProxy.report();
  }

  void counts(long g, long ng, long a) {
    total[0] += g;
    total[1] += ng;
    total[2] += a;
    if (++replies < CkNumPes()) return;
    const long want[3] = {(long)BURST * CkNumPes(), (long)BURST * CkNumNodes(),
                          (long)BURST * 2 * CkNumPes()};
    for (int i = 0; i < 3; i++)
      if (total[i] != want[i])
        CkAbort("bcast_priority: at quiescence %ld of %ld %s broadcast copies had run",
                total[i], want[i], i == 0 ? "group" : i == 1 ? "nodegroup" : "array");
    CkPrintf("bcast_priority: all tests passed\n");
    CkExit();
  }
};

class NG : public CBase_NG {
  std::mutex mu;
  std::vector<int> seq;

 public:
  std::atomic<long> counted{0};
  NG() {}
  void prio(int sender, int p) {
    std::lock_guard<std::mutex> g(mu);
    seq.push_back(p);
    if ((int)seq.size() < K) return;
    bool check = CkMyNodeSize() == 1 && CkNodeOf(sender) == CkMyNode();
    mainProxy.roundReply(!check || inOrder(seq));
    seq.clear();
  }
  void count() { counted++; }
};

class G : public CBase_G {
  std::vector<int> seq;
  long counted = 0;

  void record(int p, int sender) {
    seq.push_back(p);
    if ((int)seq.size() < K) return;
    bool check = CkMyNodeSize() == 1 && CkNodeOf(sender) == CkMyNode();
    mainProxy.roundReply(!check || inOrder(seq));
    seq.clear();
  }

 public:
  G() {}

  void hold(int kind, int sender, double secs) {
    double start = CkWallTimer();
    if (CkMyPe() == sender) {
      // a larger integer is a less urgent priority; all are behind the default 0
      for (int p = K; p >= 1; p--) {
        CkEntryOptions opts;
        opts.setPriority(p);
        if (kind == 0) {
          thisProxy.prio(p, sender, &opts);
        } else if (kind == 1) {
          PMsg *m = new (8 * sizeof(int)) PMsg;
          m->p = p;
          m->sender = sender;
          *(int *)CkPriorityPtr(m) = p;
          CkSetQueueing(m, CK_QUEUEING_IFIFO);
          thisProxy.prioMsg(m);
        } else if (kind == 2) {
          ngProxy.prio(sender, p, &opts);
        }
      }
      if (kind == 3) {
        for (int p = 1; p <= K; p++) {
          CkEntryOptions opts;
          opts.setQueueing(CK_QUEUEING_LIFO);
          thisProxy.lifo(p, sender, &opts);
        }
      }
    }
    while (CkWallTimer() - start < secs) {
    }
  }
  void prio(int p, int sender) { record(p, sender); }
  void lifo(int p, int sender) { record(-p, sender); }  // issued 1..K, expected K..1
  void prioMsg(PMsg *m) {
    record(m->p, m->sender);
    delete m;
  }

  void burst(int n) {
    for (int i = 0; i < n; i++) {
      CkEntryOptions opts;
      opts.setPriority(i % 5 - 2);
      if (i % 3 == 0) {
        thisProxy.count(&opts);
        ngProxy.count(&opts);
        aProxy.count(&opts);
      } else {
        thisProxy.count();
        ngProxy.count();
        aProxy.count();
      }
    }
  }
  void count() { counted++; }
  void report() {
    long ng = (CkMyRank() == 0) ? ngProxy.ckLocalBranch()->counted.load() : 0;
    mainProxy.counts(counted, ng, CkpvAccess(arrayCount));
  }
};

class A : public CBase_A {
 public:
  A() {}
  A(CkMigrateMessage *) {}
  void count() { CkpvAccess(arrayCount)++; }
};

static void initPe() {
  CkpvInitialize(long, arrayCount);
  CkpvAccess(arrayCount) = 0;
}

#include "bcast_priority.def.h"
