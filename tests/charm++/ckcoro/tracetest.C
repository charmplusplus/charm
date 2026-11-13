// Each element waits TRACE_ROUNDS times for a reduction it contributes to.
// Every resumed segment is a Projections execution of run(), so the logs
// hold at least TRACE_ELEMS * (TRACE_ROUNDS + 1) begin events for it
// (checked by the Makefile's trace-test). That needs every wait to suspend: a
// reduction's result cannot arrive before the element's own contribution,
// made just before the wait. (A message from a neighbour can arrive before
// run() starts, and the wait for it then returns without suspending.)
// The labelled blocks are entries "work" (once a round) and "setup" (in the
// constructor, outside any coroutine).
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
    CProxy_Stepper a = CProxy_Stepper::ckNew(TRACE_ELEMS);
    a.run();
  }
  void done() { CkExit(); }
};

class Stepper : public CBase_Stepper {
  Stepper_CORO_CODE
  int round = 0;
  double sum = 0;
  ck::coro::future step;

public:
  Stepper() {
    CK_LABEL("setup") { sum = thisIndex; }
  }
  Stepper(CkMigrateMessage* m) : CBase_Stepper(m) {}
  void pup(PUP::er& p) { p | round; p | sum; p | step; }

  ck::coroutine run() {
    CK_MIGRATABLE {
      for (round = 0; round < TRACE_ROUNDS; round++) {
        CK_LABEL("work") {
          for (int i = 1; i <= 1000; i++) sum += 1.0 / i;
        }
        contribute(CK_FUTURE_ALL(step));
        CK_WHEN(step);
      }
      contribute(CkCallback(CkReductionTarget(TraceMain, done), traceMain));
    }
  }
};

#include "tracetest.def.h"
