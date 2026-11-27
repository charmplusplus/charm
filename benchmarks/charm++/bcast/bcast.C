// Broadcast micro-benchmark: time per broadcast of an entry method with one
// int argument, for a group, a nodegroup and a chare array.
//
//   seq   PE 0 issues `batch` group broadcasts in a row, then waits until
//         every PE has run all of them; repeated until n broadcasts are done.
//   ping  the copy of a broadcast that reaches PE 0 (group), node 0
//         (nodegroup) or element 0 (array) issues the next broadcast.
//
// Usage: ./bcast [n [batch]] +pe P      (defaults 100000 1000)

#include "bcast.decl.h"

CProxy_Main mainProxy;
CProxy_G gProxy;
CProxy_NG ngProxy;
CProxy_A aProxy;
int batch;

class Main : public CBase_Main {
  int n, sent = 0, phase = 0;
  double t0;

  void report(const char *what) {
    CkPrintf("%-16s %9.1f ns per broadcast\n", what, 1e9 * (CkWallTimer() - t0) / n);
  }
  void sendBatch() {
    for (int i = 0; i < batch; i++) gProxy.seq(i);
    sent += batch;
  }

 public:
  Main(CkArgMsg *m) {
    n = m->argc > 1 ? atoi(m->argv[1]) : 100000;
    batch = m->argc > 2 ? atoi(m->argv[2]) : 1000;
    n -= n % batch;
    delete m;
    mainProxy = thisProxy;
    gProxy = CProxy_G::ckNew();
    ngProxy = CProxy_NG::ckNew();
    aProxy = CProxy_A::ckNew(CkNumPes());
    CkPrintf("bcast: %d broadcasts per phase, batch %d, %d PEs in %d processes\n", n,
             batch, CkNumPes(), CkNumNodes());
    CkStartQD(CkCallback(CkIndex_Main::start(), thisProxy));
  }
  void start() {
    t0 = CkWallTimer();
    sendBatch();
  }
  void batchDone() {
    if (sent < n) {
      sendBatch();
      return;
    }
    report("group seq");
    t0 = CkWallTimer();
    gProxy.ping(n - 1);
  }
  void pingDone() {
    switch (phase++) {
      case 0:
        report("group ping");
        t0 = CkWallTimer();
        ngProxy.ping(n - 1);
        break;
      case 1:
        report("nodegroup ping");
        t0 = CkWallTimer();
        aProxy.ping(n - 1);
        break;
      default:
        report("array ping");
        CkExit();
    }
  }
};

class G : public CBase_G {
 public:
  G() {}
  void seq(int i) {
    if (i == batch - 1) contribute(CkCallback(CkReductionTarget(Main, batchDone), mainProxy));
  }
  void ping(int left) {
    if (CkMyPe() != 0) return;
    if (left == 0) mainProxy.pingDone();
    else thisProxy.ping(left - 1);
  }
};

class NG : public CBase_NG {
 public:
  NG() {}
  void ping(int left) {
    if (CkMyNode() != 0) return;
    if (left == 0) mainProxy.pingDone();
    else thisProxy.ping(left - 1);
  }
};

class A : public CBase_A {
 public:
  A() {}
  A(CkMigrateMessage *) {}
  void ping(int left) {
    if (thisIndex != 0) return;
    if (left == 0) mainProxy.pingDone();
    else thisProxy.ping(left - 1);
  }
};

#include "bcast.def.h"
