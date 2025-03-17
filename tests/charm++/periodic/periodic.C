/*
 * This test starts a timer on the mainchare then broadcasts
 * over a group to start periodic callbacks which will print
 * from each PE every second for a total of MAX_COUNTER
 * seconds before performing a reduction to the mainchare.
 * The mainchare checks that the run indeed took MAX_COUNTER
 * seconds at minimum before exiting. If this program hangs,
 * periodic callbacks are likely not being triggered properly.
 */

#include "periodic.decl.h"

/* 
 * Change to 1 in order to test CcdCallFnAfter, otherwise
 * this tests CcdCallOnConditionKeep.
 */
#ifndef CALL_FN_AFTER
#define CALL_FN_AFTER 0
#endif

CProxy_main mProxy;
CProxy_testGroup gProxy;

static constexpr int COUNTER_MAX = 3; /* How many iters or seconds to run for */
static constexpr double TOL = 0.9;    /* 10% tolerance */

void userFn(void *arg, double time);

class main : public CBase_main {
  double startTime;
  public:

  main(CkArgMsg *msg) {
    delete msg;

    mProxy = thisProxy;
    #if CALL_FN_AFTER
    startTime = CkWallTimer();
    #else
    startTime = 0.0;
    #endif
    gProxy = CProxy_testGroup::ckNew(COUNTER_MAX);
    CkPrintf("Testing Converse periodic callbacks on %d PEs for %d seconds\n",
             CkNumPes(), COUNTER_MAX);

    gProxy.testPeriodic();
  }

  void done() {
    /* Time taken for the test should be at least MAX_COUNTER seconds */
    double totalTime = CkWallTimer() - startTime;
    if (totalTime >= ((double)COUNTER_MAX * TOL)) {
      CkPrintf("CcdPeriodic test PASSED\n");
      CkExit();
    }
    else {
      CkAbort("CcdPeriodic test FAILED: run only took %f seconds (less than minimum %d seconds)!\n",
              totalTime, COUNTER_MAX);
    }
  }
};


class testGroup : public CBase_testGroup {
  int counter;
  CkCallback cb;

  public:
  testGroup(int max) {
    CkAssert(max > 0);
    counter = max;
    cb = CkCallback(CkReductionTarget(main, done), mProxy);
  }

  void testPeriodic(void) {
#if CALL_FN_AFTER
    CcdCallFnAfter((CcdVoidFn)userFn, &counter, 1000 /*ms*/);
#else
    CcdCallOnConditionKeep(CcdPERIODIC_1s, (CcdCondFn)userFn, &counter);
#endif
  }

  void reduceToCompletion() {
    contribute(cb);
  }
};

void userFn(void *arg, double time) {
  int *counter = (int *)arg;
  (*counter)--;
  CmiPrintf("PE %d inside periodic user callback fn: counter %d, time %f\n", CkMyPe(), *counter, time);
  if (*counter == 0) {
    gProxy.ckLocalBranch()->reduceToCompletion();
  }
#if CALL_FN_AFTER
  else {
    CcdCallFnAfter((CcdVoidFn)userFn, counter, 1000 /*ms*/);
  }
#endif
}

#include "periodic.def.h"
