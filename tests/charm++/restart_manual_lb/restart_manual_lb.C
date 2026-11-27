// AtSync after a disk restart of a program that switched manual load balancing
// off again at run time.
//
// An initproc calls TurnManualLBOn(); the LBSwitch group's constructor calls
// TurnManualLBOff(), and its migration constructor does nothing, so a restart
// runs the initproc again but not the constructor. The LBManager's pup must
// carry the barrier setting across the checkpoint: otherwise the restarted run
// is in manual mode, every AtSync below is dropped, ResumeFromSync never comes
// and quiescence is reached.
//
// Elements call AtSync every step and take the next step in ResumeFromSync. The
// first run checkpoints to log/ after step -ckpt-at and exits; a run with
// +restart log must reach step -steps. Quiescence fails the test in either run.
//
// Options: -steps N (default 8), -ckpt-at K (default 4), -elems E (default 8).

#include "restart_manual_lb.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem elemProxy;
/*readonly*/ int numElems;
/*readonly*/ int numSteps;
/*readonly*/ int ckptStep;

void manualLBOn(void) { TurnManualLBOn(); }

class Main : public CBase_Main
{
 public:
  Main(CkArgMsg* m)
  {
    numSteps = 8;
    ckptStep = 4;
    numElems = 8;
    CmiGetArgIntDesc(m->argv, "-steps", &numSteps, "number of steps");
    CmiGetArgIntDesc(m->argv, "-ckpt-at", &ckptStep, "checkpoint after this step");
    CmiGetArgIntDesc(m->argv, "-elems", &numElems, "number of array elements");
    delete m;
    CkEnforce(0 < ckptStep && ckptStep < numSteps);
    mainProxy = thisProxy;
    CkStartQD(CkCallback(CkIndex_Main::quiescence(), thisProxy));
    CkPrintf("restart_manual_lb: %d elements, %d steps, checkpoint after step %d, %d PEs\n",
             numElems, numSteps, ckptStep, CkNumPes());
    CProxy_LBSwitch::ckNew();
  }

  Main(CkMigrateMessage* m) : CBase_Main(m)
  {
    mainProxy = thisProxy;
    CkStartQD(CkCallback(CkIndex_Main::quiescence(), thisProxy));
  }

  // Every PE's LBSwitch has turned manual load balancing off.
  void switchedOff()
  {
    elemProxy = CProxy_Elem::ckNew(numElems);
    elemProxy.step();
  }

  void atCheckpoint()
  {
    CkPrintf("Checkpoint after step %d\n", ckptStep);
    CkStartCheckpoint("log", CkCallback(CkIndex_Main::resumed(), thisProxy));
  }

  // After this run's checkpoint, and as the first message of a restarted run.
  void resumed()
  {
    if (!_restarted)
    {
      CkPrintf("Checkpoint written; exiting\n");
      CkExit();
    }
    CkPrintf("Restarted at step %d on %d PEs\n", ckptStep, CkNumPes());
    elemProxy.step();
  }

  void done(int stepSum)
  {
    if (!_restarted) CkAbort("reached step %d without a restart\n", numSteps);
    if (stepSum != numElems * numSteps)
      CkAbort("step sum %d, expected %d\n", stepSum, numElems * numSteps);
    CkPrintf("restart_manual_lb: PASS, step %d reached after the restart\n", numSteps);
    CkExit();
  }

  void quiescence()
  {
    CkAbort("restart_manual_lb: quiescence reached (%s run): AtSync got no ResumeFromSync\n",
            _restarted ? "restarted" : "first");
  }
};

class LBSwitch : public CBase_LBSwitch
{
 public:
  LBSwitch()
  {
    TurnManualLBOff();
    contribute(CkCallback(CkReductionTarget(Main, switchedOff), mainProxy));
  }
  LBSwitch(CkMigrateMessage* m) : CBase_LBSwitch(m) {}
};

class Elem : public CBase_Elem
{
  int it = 0;

 public:
  Elem() { usesAtSync = true; }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void pup(PUP::er& p) { p | it; }

  void step()
  {
    ++it;
    if (it == numSteps)
      contribute(sizeof(int), &it, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, done), mainProxy));
    else if (it == ckptStep && !_restarted)
      contribute(CkCallback(CkReductionTarget(Main, atCheckpoint), mainProxy));
    else
      AtSync();
  }

  void ResumeFromSync() { thisProxy[thisIndex].step(); }
};

#include "restart_manual_lb.def.h"
