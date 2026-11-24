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
//
// ---------------------------------------------------------------------------
// 2026-08-02: knobs added to diagnose the mid-run latency cliff.
//
// On Anvil at >= 2 nodes, runs intermittently step from ~2.4 us to ~200-250 us
// PER HOP partway through a phase and never recover (5 of 6 runs at 4 nodes).
// The degraded per-hop cost is the SAME at 240 and 480 PEs, which is the
// signature of a fixed per-message adder rather than congestion. Three knobs
// separate the candidate mechanisms:
//
//   -c <N>  tokens in flight (default 1). The original ring is a serialized
//           chain -- exactly ONE message outstanding -- so it measures pure
//           per-message latency and is blind to congestion. If the cliff
//           disappears at high -c, the problem is per-message wakeup/progress
//           latency, not the fabric. Total hops per phase are held constant
//           regardless of -c, so ring_ms stays comparable.
//   -s <B>  payload bytes per hop (default 0). Sweeping across the eager /
//           rendezvous threshold tests whether the cliff is messages falling
//           onto the rendezvous path.
//   -r <R>  ring laps per phase (default 100).
//   -d <ms> busy-wait on PE 0 before phase 0 (default 0). Onset was measured
//           at ~1 s of ELAPSED time in every cliffed run, near-independent of
//           payload (64x bytes) and of message rate -- the signature of a
//           timer, not a resource being consumed. This knob separates
//           "1 s after program start" from "1 s after messaging starts": if
//           the former, -d 2000 puts the cliff in phase 0.
//
// PE 0 also timestamps every pass of token 0 and prints the inter-lap deltas
// at the end. Phase-granular output cannot locate the transition: every
// cliffed run shows one INTERMEDIATE phase, so the switch happens mid-phase.
// Laps per phase = R/c, contiguous in print order, so the series segments by
// index offline.
#include "qdbench.decl.h"
#include <vector>

CProxy_Main mainProxy;
CProxy_Ring ringProxy;

static const int NPHASES = 10;

class Main : public CBase_Main {
  int phase = 0;
  double t0 = 0.0;
  double settle[NPHASES];
  double work[NPHASES];
  double compute[NPHASES];
  double t_start = 0.0;
  int done = 0;       // tokens finished this phase
  int conc = 1;       // -c
  int payload = 0;    // -s
  int rounds = 100;   // -r
  std::vector<char> buf;

public:
  Main(CkArgMsg* m) {
    int delayMs = 0;
    CmiGetArgInt(m->argv, "-c", &conc);
    CmiGetArgInt(m->argv, "-s", &payload);
    CmiGetArgInt(m->argv, "-r", &rounds);
    CmiGetArgInt(m->argv, "-d", &delayMs);
    delete m;
    if (conc < 1) conc = 1;
    if (payload < 0) payload = 0;
    mainProxy = thisProxy;
    ringProxy = CProxy_Ring::ckNew();
    buf.assign(payload, 'x');
    CkPrintf("qdbench: %d PEs, %d processes, %d phases, %d ring laps/phase, "
             "conc %d, payload %d B, predelay %d ms\n",
             CkNumPes(), CkNumNodes(), NPHASES, rounds, conc, payload, delayMs);
    if (delayMs > 0) {
      // Deliberately a busy-wait on PE 0, not a sleep: the other PEs sit idle
      // in the scheduler, which is the state we want to age.
      double until = CkWallTimer() + delayMs / 1000.0;
      while (CkWallTimer() < until) { }
    }
    startPhase();
  }

  void startPhase() {
    // Local-compute control, timed per phase and touching no messaging at all.
    // The degradation is elapsed-time-triggered and survives a pure busy-wait,
    // so the two candidate families are (a) something in the transport and
    // (b) something machine-wide such as CPU frequency / power-budget decay
    // under 480 spinning PEs. If compute_ms stays flat while ring_ms blows up,
    // it is the transport; if both degrade together, it is the machine.
    double c0 = CkWallTimer();
    volatile double acc = 0.0;
    for (int i = 1; i <= 3000000; i++) acc += 1.0 / i;
    compute[phase] = CkWallTimer() - c0;
    t_start = CkWallTimer();
    done = 0;
    // Total hops held constant across -c so ring_ms stays comparable; the
    // remainder goes to the first tokens.
    const int total = rounds * CkNumPes();
    for (int i = 0; i < conc; i++) {
      int hops = total / conc + (i < total % conc ? 1 : 0);
      ringProxy[0].token(hops, i, payload, buf.data());
    }
  }

  void tokenDone() {
    if (++done < conc) return;
    // Every token has been consumed; the machine is quiet except for this
    // method. Timestamp, then ask for quiescence.
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
      CkPrintf("phase  ring_ms  qd_settle_ms  compute_ms\n");
      for (int i = 0; i < NPHASES; i++)
        CkPrintf("%5d %8.3f %13.3f %11.3f\n", i, work[i] * 1e3, settle[i] * 1e3, compute[i] * 1e3);
      double s = 0;
      for (int i = 1; i < NPHASES; i++) s += settle[i]; // skip warmup phase 0
      CkPrintf("QDBENCH mean_settle_ms %.3f (phases 1-%d)\n",
               s / (NPHASES - 1) * 1e3, NPHASES - 1);
      ringProxy[0].reportLaps();
    }
  }

  void lapsPrinted() { CkExit(); }
};

class Ring : public CBase_Ring {
  std::vector<double> laps; // PE 0 only, token 0 only

public:
  Ring() {}

  void token(int hops, int tokenId, int n, char payload[]) {
    if (CkMyPe() == 0 && tokenId == 0) laps.push_back(CkWallTimer());
    if (hops == 0) {
      mainProxy.tokenDone();
      return;
    }
    ringProxy[(CkMyPe() + 1) % CkNumPes()].token(hops - 1, tokenId, n, payload);
  }

  void reportLaps() {
    // Inter-lap deltas in ms. Laps per phase = rounds/conc, contiguous in
    // order, so the cliff onset lap segments by index offline.
    CkPrintf("QDBENCH lap_deltas_ms n=%zu\n", laps.size() > 0 ? laps.size() - 1 : 0);
    for (size_t i = 1; i < laps.size(); i++) {
      CkPrintf("%.4f%s", (laps[i] - laps[i - 1]) * 1e3,
               (i % 20 == 0 || i == laps.size() - 1) ? "\n" : " ");
    }
    CkPrintf("QDBENCH lap_deltas_end\n");
    mainProxy.lapsPrinted();
  }
};

#include "qdbench.def.h"
