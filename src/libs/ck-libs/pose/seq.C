// File: seq.C
// Module for sequential simulation strategy class
#include "pose.h"

void seq::Step()
{
  Event *ev;
  // Prepare to execute an event
  ev = eq->currentPtr;
  currentEvent = ev;
  if (ev->timestamp < POSE_GlobalTS)
    CkPrintf("WARNING: SEQUENTIAL POSE BUG! Event timestamp %d is less than a previous one! This is due to stupid Charm++ Scheduler implementation and needs to be fixed.\n", ev->timestamp);
  POSE_GlobalTS = ev->timestamp;
  parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
  if (userObj->OVT() > POSE_GlobalClock)
    POSE_GlobalClock = userObj->OVT();
  ev->done = 1;
  eq->ShiftEvent();                       // move on to next event
  eq->CommitAll(parent);
}
