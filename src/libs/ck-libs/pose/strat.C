// File: strat.C
// Module for basic simulation strategy class for protocols such as 
// optimistic and conservative.
// Last Modified: 06.05.01 by Terry L. Wilmarth

#include "pose.h"

strat::strat() // basic initialization constructor
{
  eq = NULL;
  userObj = NULL;
  parent = NULL;
  currentEvent = targetEvent = RBevent = NULL;
  voted = 0;
  localPVT = (PVT *)CkLocalBranch(ThePVT);
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
}

// initialize parent sim object pointers
void strat::init(eventQueue *q, rep *obj, sim *p, int pIdx)
{
  eq = q;
  userObj = obj;
  parent = p;
  parentIdx = pIdx;
  initSync();
}

// Strategy specific forward execution step.
// Code here MUST be overridden, but this gives a basic idea of how
// a forward execution step should go.  Strategies must determine if it is
// safe to execute an event.
void strat::Step()
{
  Event *ev;

  parent->Deactivate();        // since this is executing, there is no longer a
                               // queued Step message, so deactivate
  ev = eq->currentPtr;         // get event to execute
  currentEvent = ev;           // set currentEvent
  if (ev->timestamp >= 0) {    // make sure it's not the back sentinel
    parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
    ev->done = 1;              // mark it done
    eq->ShiftEvent();          // shift to next event
    // reactivate and re-enqueue Step message
    parent->Activate();
    POSE_Objects[parentIdx].Step();
  }
}
