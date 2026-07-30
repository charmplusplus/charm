// Quiescence-detection test: repeated CkStartQD WITHOUT program exit.
//
// Each phase runs a message ring over all PEs (real inter-PE and, in
// multi-process runs, inter-process traffic), then quiesces via CkStartQD
// with a callback; the next phase starts from the callback. 10 phases.
//
// This exercises what one-shot QD-at-exit uses cannot: QD must detect
// quiescence, deliver its callback, and then correctly observe the NEW
// activity of the following phase, ten times in a row. A hang means QD
// state is not resetting between detections.
//
// The printed qd_settle_ms is also a performance canary: the time from the
// last application message being processed to the QD callback firing, on an
// otherwise idle machine. On a healthy runtime this is well under a
// millisecond at small scale (QD is ~3 rounds of counting over a spanning
// tree of asynchronous messages). Settles of tens of milliseconds mean the
// transport is delaying sparse single messages (observed on reconverse/LCI,
// 2026-07: ~12.7 ms per idle-path cross-process hop on InfiniBand and
// ~5-10 ms on ofi/tcp, while classic converse showed ~0.5 ms settles in the
// same 2-process configuration).
#include "qdbench.decl.h"

CProxy_Main mainProxy;
CProxy_Ring ringProxy;

static const int NPHASES = 10;
static const int ROUNDS = 100; // ring laps per phase

class Main : public CBase_Main {
  int phase = 0;
  double t0 = 0.0;
  double settle[NPHASES];
  double work[NPHASES];
  double t_start = 0.0;

public:
  Main(CkArgMsg* m) {
    delete m;
    mainProxy = thisProxy;
    ringProxy = CProxy_Ring::ckNew();
    CkPrintf("qdbench: %d PEs, %d processes, %d phases, %d ring laps/phase\n",
             CkNumPes(), CkNumNodes(), NPHASES, ROUNDS);
    startPhase();
  }

  void startPhase() {
    t_start = CkWallTimer();
    ringProxy[0].token(ROUNDS * CkNumPes());
  }

  void phaseDone() {
    // Last ring message has been processed; the machine is quiet except for
    // this method. Timestamp, then ask for quiescence.
    t0 = CkWallTimer();
    work[phase] = t0 - t_start;
    CkStartQD(CkCallback(CkIndex_Main::qdReached(), mainProxy));
  }

  void qdReached() {
    settle[phase] = CkWallTimer() - t0;
    phase++;
    if (phase < NPHASES) {
      startPhase();
    } else {
      CkPrintf("phase  ring_ms  qd_settle_ms\n");
      for (int i = 0; i < NPHASES; i++)
        CkPrintf("%5d %8.3f %13.3f\n", i, work[i] * 1e3, settle[i] * 1e3);
      double s = 0;
      for (int i = 1; i < NPHASES; i++) s += settle[i]; // skip warmup phase 0
      CkPrintf("QDBENCH mean_settle_ms %.3f (phases 1-%d)\n",
               s / (NPHASES - 1) * 1e3, NPHASES - 1);
      CkExit();
    }
  }
};

class Ring : public CBase_Ring {
public:
  Ring() {}
  void token(int hops) {
    if (hops == 0) {
      mainProxy.phaseDone();
      return;
    }
    ringProxy[(CkMyPe() + 1) % CkNumPes()].token(hops - 1);
  }
};

#include "qdbench.def.h"
