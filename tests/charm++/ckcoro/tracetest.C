// Each element waits TRACE_ROUNDS times for a ping from its left neighbour.
// Every resumed segment is a Projections execution of run(), so the logs
// hold TRACE_ELEMS * (TRACE_ROUNDS + 1) begin events for it (checked by
// the Makefile's trace-test).
#include "tracetest.decl.h"
#include "ckcoro.h"

#define TRACE_ELEMS 4
#define TRACE_ROUNDS 10

/* readonly */ CProxy_TraceMain traceMain;

class TraceMain : public CBase_TraceMain {
public:
  TraceMain(CkArgMsg* m) {
    delete m;
    traceMain = thisProxy;
    CProxy_Pinger a = CProxy_Pinger::ckNew(TRACE_ELEMS);
    a.run();
  }
  void done() { CkExit(); }
};

class Pinger : public CBase_Pinger {
  Pinger_CORO_CODE
  int round = 0;

public:
  Pinger() {}
  Pinger(CkMigrateMessage* m) : CBase_Pinger(m) {}
  void pup(PUP::er& p) { p | round; }

  ck::coroutine run() {
    CK_MIGRATABLE {
      for (round = 0; round < TRACE_ROUNDS; round++) {
        thisProxy[(thisIndex + 1) % TRACE_ELEMS].ping(round);
        CK_WHEN(inbox::ping(round), ck::coro::skip);
      }
      contribute(CkCallback(CkReductionTarget(TraceMain, done), traceMain));
    }
  }
};

#include "tracetest.def.h"
