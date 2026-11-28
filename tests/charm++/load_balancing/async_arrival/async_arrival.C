// Load balancing steps through AtSync and through the split barrier
// (AtSyncStart / AtSyncWait), with and without +LBAsync, over several steps.
//
//   ./async_arrival <steps> <split> <wait> [+LBAsync] [+balancer ...]
//
// split = 0: each step is AtSync(wait).
// split = 1: each step is AtSyncStart(wait); the element keeps receiving
//            messages, polls AtSyncStepInFlight(), and calls AtSyncWait()
//            once the step is over.
// wait  = 0: the balancer does not hold its migration barrier for the arrival
//            (an asynchronous arrival). CentralLB counts these arrivals
//            separately and withholds the end of the step -- and so the next
//            step -- until every one of them is in, so this needs at least two
//            steps to fail.
//
// Every element must be resumed exactly once per step: the step can finish
// inside the AtSyncStart call that completes a PE's barrier (with no balancer,
// for one), and AtSyncWait must not resume the element a second time.
//
// The balancer has to move objects for the arrival paths to run, so each
// element reports a load that moves the hot spot every step; a pattern that
// merely permutes equal loads leaves the mapping optimal and nothing migrates.

#include "async_arrival.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_elements;

#define OBJS_PER_PE 4

static void retryWait(void* arg, double)
{
  auto* cb = static_cast<CkCallback*>(arg);
  cb->send();
  delete cb;
}

class Main : public CBase_Main
{
  int step = 0, num_steps;
  CProxy_Blk blocks;

  static void watchdog(void* arg, double)
  {
    Main* m = (Main*)arg;
    CkAbort("async_arrival: hung in load balancing step %d of %d", m->step + 1,
            m->num_steps);
  }

public:
  Main(CkArgMsg* msg)
  {
    num_steps = (msg->argc > 1) ? atoi(msg->argv[1]) : 4;
    const int split = (msg->argc > 2) ? atoi(msg->argv[2]) : 0;
    const int wait = (msg->argc > 3) ? atoi(msg->argv[3]) : 0;
    delete msg;

    mainProxy = thisProxy;
    num_elements = CkNumPes() * OBJS_PER_PE;
    blocks = CProxy_Blk::ckNew(split, wait, num_elements);
    CkPrintf("[TEST] %d elements, %d steps, split=%d wait=%d\n", num_elements, num_steps,
             split, wait);
    CcdCallFnAfter(watchdog, this, 60000);
    blocks.run();
  }

  // Driven from a reduction, so that a step boundary means "every element has
  // resumed", not "this one has".
  void stepDone(int maxResumes, int negMinResumes)
  {
    step++;
    if (maxResumes != step || -negMinResumes != step)
      CkAbort("async_arrival: after step %d elements were resumed between %d and %d times",
              step, -negMinResumes, maxResumes);
    CkPrintf("[TEST] step %d of %d complete\n", step, num_steps);
    if (step == num_steps)
    {
      CkPrintf("[TEST] PASS\n");
      CkExit();
    }
    else
    {
      blocks.run();
    }
  }
};

class Blk : public CBase_Blk
{
  int split, wait;
  int stamp;  // must survive migration unchanged
  int resumes;

public:
  Blk(int split_, int wait_) : split(split_), wait(wait_), stamp(thisIndex * 7 + 3), resumes(0)
  {
    usesAtSync = true;
    usesAutoMeasure = false;
  }
  Blk(CkMigrateMessage* m) : CBase_Blk(m) {}

  void run()
  {
    if (!split)
    {
      AtSync(wait);
      return;
    }
    if (AtSyncStart(wait) == AtSyncStatus::Blocked) return;
    thisProxy[thisIndex].waitForStep();
  }

  void waitForStep()
  {
    if (AtSyncStepInFlight())
    {
      // Through the proxy: the element may migrate before this fires.
      CcdCallFnAfter(retryWait,
                     new CkCallback(CkIndex_Blk::waitForStep(), thisProxy[thisIndex]), 1);
      return;
    }
    AtSyncWait();
  }

  void UserSetLBLoad() override
  {
    const bool hot = (thisIndex == (resumes % num_elements));
    setObjTime(hot ? 100.0 : 1.0);
  }

  void ResumeFromSync() override
  {
    if (stamp != thisIndex * 7 + 3)
      CkAbort("async_arrival: element %d state corrupted across migration", thisIndex);
    resumes++;
    int r[2] = {resumes, -resumes};
    contribute(2 * sizeof(int), r, CkReduction::max_int,
               CkCallback(CkReductionTarget(Main, stepDone), mainProxy));
  }

  void pup(PUP::er& p) override
  {
    CBase_Blk::pup(p);
    p | split;
    p | wait;
    p | stamp;
    p | resumes;
  }
};

#include "async_arrival.def.h"
