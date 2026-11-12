/*
 * Heat diffusion on a ring of 1D strips, with coroutine entry methods.
 *
 *   ./heat1d [strips [cells [iterations [reportFreq [lbFreq]]]]] +pN [+balancer RotateLB]
 *
 * Each Strip holds cellsPerStrip cells; its two boundary cells go to its
 * neighbours every iteration, tagged with the iteration number, and the
 * strip waits for both neighbours' values (a conjunction) before it
 * updates. Every reportFreq iterations the strips reduce the largest
 * change to Main, whose own coroutine waits for the reports; at the end
 * they reduce an integer checksum of the temperatures (sum of squares,
 * truncated per cell), which is the same
 * on any number of PEs, with migration (+balancer RotateLB, lbFreq > 0)
 * and under +ck_fakemigrate.
 */
#include <vector>
#include <cstdlib>
#include "heat1d.decl.h"
#include "ckcoro.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int numStrips;
/* readonly */ int cellsPerStrip;
/* readonly */ int maxIter;
/* readonly */ int reportFreq;
/* readonly */ int lbFreq;

class Main : public CBase_Main {
  Main_CORO_CODE
  int reports;   // live across waits: a member

 public:
  Main(CkArgMsg *m) {
    numStrips = m->argc > 1 ? atoi(m->argv[1]) : 8;
    cellsPerStrip = m->argc > 2 ? atoi(m->argv[2]) : 64;
    maxIter = m->argc > 3 ? atoi(m->argv[3]) : 200;
    reportFreq = m->argc > 4 ? atoi(m->argv[4]) : 50;
    lbFreq = m->argc > 5 ? atoi(m->argv[5]) : 0;
    delete m;
    mainProxy = thisProxy;
    CkPrintf("heat1d: %d strips of %d cells, %d iterations, report every %d, LB every %d\n",
             numStrips, cellsPerStrip, maxIter, reportFreq, lbFreq);
    CProxy_Strip strips = CProxy_Strip::ckNew(numStrips);
    strips.run();
    thisProxy.run();
  }

  // The driver as a coroutine: each report is a reduction delivered to the
  // inbox entry report(); the block form gives its parameter a block scope.
  ck::coroutine run() {
    CK_MIGRATABLE {
      for (reports = 1; reports <= maxIter / reportFreq; reports++) {
        CK_WHEN_BLOCK(inbox::report(), maxChange) {
          CkPrintf("iteration %d: largest change %.6e\n", reports * reportFreq, maxChange);
        }
      }
      CK_WHEN_BLOCK(inbox::done(), checksum) {
        CkPrintf("heat1d: %d iterations, checksum %lld\n", maxIter, checksum);
      }
      CkExit();
    }
  }
};

class Strip : public CBase_Strip {
  Strip_CORO_CODE
  std::vector<double> t, next;   // cells 1..n, with the two ghosts at 0 and n+1
  int iter;                      // live across waits: a member, pupped
  double leftGhost, rightGhost;  // received at the wait: members, pupped
  double maxChange;

 public:
  Strip() : t(cellsPerStrip + 2, 0.0), next(cellsPerStrip + 2, 0.0), iter(0), maxChange(0) {
    // A hot spot in the middle of strip 0.
    if (thisIndex == 0) t[cellsPerStrip / 2] = 1000.0;
    usesAtSync = true;
  }
  Strip(CkMigrateMessage *m) : CBase_Strip(m) {}

  void pup(PUP::er &p) {
    p | t; p | next; p | iter; p | leftGhost; p | rightGhost; p | maxChange;
  }

  ck::coroutine run() {
    CK_MIGRATABLE {
      for (iter = 1; iter <= maxIter; iter++) {
        if (lbFreq > 0 && iter % lbFreq == 0) {
          AtSync();                            // the element may migrate while it waits here
          CK_WHEN(inbox::ResumeFromSync());
        }
        // Boundary cells to the neighbours on the ring, tagged with the iteration.
        thisProxy[(thisIndex + numStrips - 1) % numStrips].fromRight(iter, t[1]);
        thisProxy[(thisIndex + 1) % numStrips].fromLeft(iter, t[cellsPerStrip]);
        // Both neighbours' values for this iteration (when fromLeft[iter](...), fromRight[iter](...)).
        CK_WHEN_ALL(CK_ALT(inbox::fromLeft(iter), ck::coro::skip, leftGhost),
                    CK_ALT(inbox::fromRight(iter), ck::coro::skip, rightGhost));
        update();
        if (iter % reportFreq == 0)
          contribute(sizeof(double), &maxChange, CkReduction::max_double,
                     CkCallback(CkReductionTarget(Main, report), mainProxy));
      }
      {
        // Sum of squares (the sum itself is conserved by the diffusion).
        long long sum = 0;
        for (int i = 1; i <= cellsPerStrip; i++) sum += (long long)(t[i] * t[i] * 1e3);
        contribute(sizeof(sum), &sum, CkReduction::sum_long_long,
                   CkCallback(CkReductionTarget(Main, done), mainProxy));
      }
    }
  }

  void update() {
    t[0] = leftGhost;
    t[cellsPerStrip + 1] = rightGhost;
    maxChange = 0;
    for (int i = 1; i <= cellsPerStrip; i++) {
      next[i] = t[i] + 0.25 * (t[i - 1] - 2 * t[i] + t[i + 1]);
      double change = next[i] - t[i];
      if (change < 0) change = -change;
      if (change > maxChange) maxChange = change;
    }
    t.swap(next);
  }
};

#include "heat1d.def.h"
