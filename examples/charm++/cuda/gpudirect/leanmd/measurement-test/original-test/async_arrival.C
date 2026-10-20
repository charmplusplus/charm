// Exercises asynchronous migration -- AtSync(0), where the load balancer does
// not hold its migration barrier for the arrival -- across several balancing
// steps.
//
// CentralLB counts these arrivals separately, in future_migrates_completed, and
// CheckMigrationComplete withholds its second token until every one of them has
// landed. That token is what releases DoneRegisteringObjects and so the next
// step. If arrivals are not counted, the first step appears to succeed and the
// second one never starts, so this test needs at least two steps to fail.
//
// The balancer must be a CentralLB (TreeLB rejects async arrivals outright),
// and it has to actually move objects, so each element reports a load that
// rotates with the step:
//   ./async_arrival 4 +p4 +balancer GreedyCentralLB

#include "async_arrival.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_steps;
/*readonly*/ int num_elements;

#define OBJS_PER_PE 4

class Main : public CBase_Main
{
private:
  int step;
  CProxy_Blk blocks;

public:
  Main(CkArgMsg* msg)
  {
    num_steps = (msg->argc > 1) ? atoi(msg->argv[1]) : 4;
    delete msg;

    mainProxy = thisProxy;
    step = 0;
    num_elements = CkNumPes() * OBJS_PER_PE;
    blocks = CProxy_Blk::ckNew(num_elements);
    CkPrintf("[TEST] %d elements, %d steps\n", CkNumPes() * OBJS_PER_PE, num_steps);
    blocks.run();
  }

  // Driven from the reduction rather than from each element, so that a step
  // boundary means "every element has resumed", not "this one has".
  void stepDone()
  {
    step++;
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
private:
  int stamp;  // must survive migration unchanged
  int resumes;

public:
  Blk() : stamp(thisIndex * 7 + 3), resumes(0)
  {
    usesAtSync = true;
    usesAutoMeasure = false;
  }
  Blk(CkMigrateMessage* m) : CBase_Blk(m) {}

  void run() { AtSync(0); }

  // Move the hot spot to a different element every step, so a greedy strategy
  // has to reshuffle each time. A pattern that merely permutes equal loads
  // leaves the existing mapping optimal and nothing migrates, and then this
  // test proves nothing.
  void UserSetLBLoad() override
  {
    const bool hot = (thisIndex == (resumes % num_elements));
    setObjTime(hot ? 100.0 : 1.0);
  }

  void ResumeFromSync() override
  {
    if (stamp != thisIndex * 7 + 3)
      CkAbort("[TEST] element %d state corrupted across migration\n", thisIndex);
    resumes++;
    contribute(CkCallback(CkReductionTarget(Main, stepDone), mainProxy));
  }

  void pup(PUP::er& p) override
  {
    CBase_Blk::pup(p);
    p | stamp;
    p | resumes;
  }
};

#include "async_arrival.def.h"
