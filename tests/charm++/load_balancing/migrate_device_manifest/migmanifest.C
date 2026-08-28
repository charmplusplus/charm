// Manifest direction-mismatch test for device migration.
//
// Every element carries two device buffers pupped with pup_buffer_device.
// Run with a migrating balancer (RotateLB) and the staged transport forced,
// so the per-buffer manifest is recorded and checked:
//
//   (no injection)                 -> completes, prints "TEST PASSED"
//   MIGTEST_INJECT_MISMATCH=1      -> unpack asks for a wrong size on the
//                                     second buffer; the run MUST abort with
//                                     "Device pup direction mismatch"
//   MIGTEST_INJECT_MISSING=1       -> unpack skips the second buffer; the
//                                     run MUST abort with the consumed-count
//                                     mismatch
//
// The aborts are the expected outcome of the injection runs: this test
// passes by dying with the right message.

#include "migmanifest.decl.h"
#include "hapi.h"
#include <cstdlib>

CProxy_Main mainProxy;
int nElems;

static const int N_A = 1000;  // doubles in the first buffer
static const int N_B = 2000;  // doubles in the second buffer
static const int ROUNDS = 4;

class Main : public CBase_Main {
  CProxy_Elem elems;
  int doneCount;
  int round;

 public:
  Main(CkArgMsg* m) {
    delete m;
    doneCount = 0;
    round = 0;
    nElems = 8;
    mainProxy = thisProxy;
    elems = CProxy_Elem::ckNew(nElems);
    elems.iterate();
  }

  void done() {
    if (++doneCount < nElems) return;
    doneCount = 0;
    if (++round == ROUNDS) {
      CkPrintf("TEST PASSED (%d rounds of migration, no mismatch injected)\n",
               ROUNDS);
      CkExit();
    } else {
      elems.iterate();
    }
  }
};

class Elem : public CBase_Elem {
 public:
  double* d_a;
  double* d_b;

  Elem() {
    usesAtSync = true;
    hapiCheck(hapiMalloc((void**)&d_a, N_A * sizeof(double)));
    hapiCheck(hapiMalloc((void**)&d_b, N_B * sizeof(double)));
  }

  Elem(CkMigrateMessage* m) : CBase_Elem(m) {
    usesAtSync = true;
    d_a = NULL;
    d_b = NULL;
  }

  ~Elem() {
    // Arena-interior after a migration, ordinary allocations before one:
    // hapiFreeMigratable handles both.
    if (d_a) hapiFreeMigratable(d_a);
    if (d_b) hapiFreeMigratable(d_b);
  }

  void pup(PUP::er& p) {
    static const bool injectMismatch =
        (getenv("MIGTEST_INJECT_MISMATCH") != NULL);
    static const bool injectMissing =
        (getenv("MIGTEST_INJECT_MISSING") != NULL);

    p.pup_buffer_device(d_a, N_A);

    // The injections lie only on the unpack side, which is exactly the
    // direction mismatch the manifest exists to catch.
    if (p.isUnpacking() && injectMissing) return;
    if (p.isUnpacking() && injectMismatch)
      p.pup_buffer_device(d_b, N_B - 500);
    else
      p.pup_buffer_device(d_b, N_B);
  }

  void iterate() {
    // Deliberate load imbalance (element i carries ~(i+1) units of walltime)
    // so central greedy balancers produce real cross-PE migrations -- this
    // app doubles as the exerciser for the memory contract's batched
    // execution protocol, which only central balancers run.
    const double until = CkWallTimer() + 0.0005 * (thisIndex + 1);
    while (CkWallTimer() < until) { }
    AtSync();
  }

  void ResumeFromSync() { mainProxy.done(); }
};

#include "migmanifest.def.h"
