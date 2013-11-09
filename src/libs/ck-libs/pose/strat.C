/// Simulation synchronization strategy base class
#include "pose.h"

/// Basic Constructor
strat::strat() 
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  eq = NULL;
  userObj = NULL;
  parent = NULL;
  currentEvent = targetEvent = NULL;
#ifndef SEQUENTIAL_POSE
  localPVT = (PVT *)CkLocalBranch(ThePVT);
#endif
  STRAT_T = INIT_T;
}

/// Initializer
void strat::init(eventQueue *q, rep *obj, sim *p, int pIdx)
{
  eq = q;  userObj = obj;  parent = p;  initSync();
}

/// Strategy-specific forward execution step
void strat::Step()
{
  Event *ev;
  parent->Deactivate(); // since this is executing, there is no longer a
  // queued Step message, so deactivate
  ev = eq->currentPtr; // get event to execute
  currentEvent = ev; // set currentEvent
  if (ev->timestamp >= 0) { // make sure it's not the back sentinel
    ev->done = 2; // mark it executing
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // mark it done
    eq->ShiftEvent(); // shift to next event
    parent->Activate();
    POSE_Objects[parent->thisIndex].Step();
  }
}
