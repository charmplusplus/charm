#ifndef __PROFILE_H__
#define __PROFILE_H__

// Stage 0 instrumentation for GPU_RESIDENT_PLAN.md. Per-PE phase timers and
// remote-request counters, carried by the DataManager -- a group, so one
// instance per PE. Timestamps are always taken (CmiWallTimer() is a clock
// read, and there are about a dozen phase boundaries per iteration); only the
// report at the end of the run is gated, on BARNES_PHASE_REPORT.
//
// The number this exists to produce is the traversal line: the wall span of
// the traversal phase minus the time actually spent executing walk code and
// blocked on the device. What is left is what the tree pieces spent waiting
// for remote data. The plan orders its stages on whether that dominates, and
// right now that is a suspicion rather than a measurement.

#include "charm++.h"

// Defined in TreePiece.cpp, where the walks run. thread_local because in this
// build PEs are threads in a process, so a plain file-scope double would
// aggregate the whole process and hide the per-PE distribution.
extern thread_local double _tpWalkLocal;
extern thread_local double _tpWalkRemote;
extern thread_local double _tpGpuFlush;

struct PhaseProfile {
  enum Phase {
    KEYGEN = 0,   // hashParticleCoordinates
    SORT_PRE,     // quickSort feeding the histogram
    HIST,         // every histogram round: a span, so it counts the round trips
    DISTRIB,      // particle exchange: a span
    SORT_POST,    // quickSort of the concatenated submissions
    BUILD,        // buildTree
    UPLOAD,       // gpuParticles.upload, the H2D of the particles
    MOMENTS,      // makeMoments through treeReady: a span
    TRAV,         // startTraversal through the last traversalsDone: a span
    FINISH,       // finishIteration through the dt reduction
    ADVANCE,      // kickDriftKick and the iteration teardown
    NUM_PHASES
  };

  static const char *name(int p){
    static const char *n[NUM_PHASES] = {
      "keygen", "sort-pre", "histogram", "distribute", "sort-post",
      "buildtree", "h2d-upload", "moments", "traversal", "finish", "advance"
    };
    return n[p];
  }

  // A phase is either inline (begin/end in one function) or a span whose ends
  // are in different entry methods. Both use the same pair; a span simply
  // stays open across messages. open < 0 means not currently open, which makes
  // a stray end() a no-op rather than a wild accumulation.
  double acc[NUM_PHASES];
  double open[NUM_PHASES];

  // Remote data requests. "sent" counts misses that actually put a message on
  // the wire -- a second requestor for the same key joins the outstanding
  // request instead of sending again. "served" counts requests this PE
  // answered on someone else's behalf.
  CmiUInt8 nodeReqSent, partReqSent;
  CmiUInt8 nodeReqServed, partReqServed;
  CmiUInt8 nodeReplies, partReplies;
  double nodeReqLatency, partReqLatency;

  int iterations;

  PhaseProfile(){
    for(int i = 0; i < NUM_PHASES; i++){ acc[i] = 0.0; open[i] = -1.0; }
    nodeReqSent = partReqSent = 0;
    nodeReqServed = partReqServed = 0;
    nodeReplies = partReplies = 0;
    nodeReqLatency = partReqLatency = 0.0;
    iterations = 0;
  }

  void begin(int p){ open[p] = CmiWallTimer(); }

  void end(int p){
    if(open[p] < 0.0) return;
    acc[p] += CmiWallTimer() - open[p];
    open[p] = -1.0;
  }

  // For a span that ends where the next one begins.
  void handoff(int from, int to){ end(from); begin(to); }

  static bool enabled(){ return getenv("BARNES_PHASE_REPORT") != NULL; }

  void report() const {
    if(!enabled()) return;

    double total = 0.0;
    for(int i = 0; i < NUM_PHASES; i++) total += acc[i];

    CkPrintf("[PHASE] pe %2d  %d iterations, %.4f s accounted\n",
             CkMyPe(), iterations, total);
    for(int i = 0; i < NUM_PHASES; i++){
      CkPrintf("[PHASE] pe %2d  %-11s %8.4f s  %5.1f%%  %8.4f ms/iter\n",
               CkMyPe(), name(i), acc[i],
               total > 0.0 ? 100.0*acc[i]/total : 0.0,
               iterations > 0 ? 1000.0*acc[i]/iterations : 0.0);
    }

    // The line the stage ordering turns on. The walk timers already include
    // the device flush, since flushGpu() is called from inside the walk entry
    // methods, so the flush is subtracted out separately rather than added.
    const double span = acc[TRAV];
    const double walk = _tpWalkLocal + _tpWalkRemote;
    const double blocked = span - walk;
    CkPrintf("[PHASE] pe %2d  traversal span %.4f s = walk %.4f s "
             "(local %.4f + remote %.4f, of which device flush %.4f) "
             "+ blocked %.4f s (%.1f%%)\n",
             CkMyPe(), span, walk, _tpWalkLocal, _tpWalkRemote, _tpGpuFlush,
             blocked, span > 0.0 ? 100.0*blocked/span : 0.0);

    CkPrintf("[PHASE] pe %2d  node reqs sent %llu served %llu replies %llu "
             "avg latency %.1f us\n",
             CkMyPe(), nodeReqSent, nodeReqServed, nodeReplies,
             nodeReplies > 0 ? 1e6*nodeReqLatency/nodeReplies : 0.0);
    CkPrintf("[PHASE] pe %2d  part reqs sent %llu served %llu replies %llu "
             "avg latency %.1f us\n",
             CkMyPe(), partReqSent, partReqServed, partReplies,
             partReplies > 0 ? 1e6*partReqLatency/partReplies : 0.0);
    CkPrintf("[PHASE] pe %2d  remote round trips per iteration: %.0f node, "
             "%.0f particle\n",
             CkMyPe(),
             iterations > 0 ? (double)nodeReqSent/iterations : 0.0,
             iterations > 0 ? (double)partReqSent/iterations : 0.0);
  }
};

#endif
