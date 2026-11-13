#include "phaselib.decl.h"
#include "phaselib.h"
#include "mpi-interoperate.h"

CProxy_PhaseMain phaseMain;
CProxy_PhaseElem phaseArray;
int phaseNumElems;
static long lastResult = -1;

long PhaseExpected(int iter) {
  return (long)iter * phaseNumElems * (phaseNumElems + 1) / 2;
}

class PhaseMain : public CBase_PhaseMain {
public:
  PhaseMain(CkArgMsg *m) {
    delete m;
    phaseNumElems = 4 * CkNumPes();
    phaseMain = thisProxy;
    phaseArray = CProxy_PhaseElem::ckNew(phaseNumElems);
    // End the CharmLibInit phase once array creation has completed everywhere.
    CkStartQD(CkCallback(CkIndex_PhaseMain::initDone(), thisProxy));
  }
  void initDone() { CkExit(); }
  void run(int iter) {
    currentIter = iter;
    phaseArray.work(iter);
  }
  void done(long sum) {
    lastResult = sum;
    CkPrintf("Charm phase %d: reduction sum %ld (expected %ld) on %d PEs\n",
             currentIter, sum, PhaseExpected(currentIter), CkNumPes());
    CkExit();
  }
private:
  int currentIter = 0;
};

class PhaseElem : public CBase_PhaseElem {
public:
  PhaseElem() {}
  PhaseElem(CkMigrateMessage *) {}
  void work(int iter) {
    long v = (long)iter * (thisIndex + 1);
    contribute(sizeof(v), &v, CkReduction::sum_long,
               CkCallback(CkReductionTarget(PhaseMain, done), phaseMain));
  }
  void abortHere() {
    CkAbort("mpi_interop: deliberate abort on PE %d (process %d of %d)", CkMyPe(),
            CkMyNode(), CkNumNodes());
  }
};

void PhaseRun(int iter) {
  if (CkMyPe() == 0) phaseMain.run(iter);
  StartCharmScheduler();
}

void PhaseAbort() {
  if (CkMyPe() == 0) phaseArray[phaseNumElems - 1].abortHere();
  StartCharmScheduler();
}

long PhaseResult() { return lastResult; }

#include "phaselib.def.h"
