// File: seq.C
// Module for sequential simulation strategy class
#include "pose.h"

void seq::Step()
{
  Event *ev;
  // Prepare to execute an event
  ev = eq->currentPtr;
  currentEvent = ev;
  parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
  if (userObj->OVT() > POSE_GlobalClock)
    POSE_GlobalClock = userObj->OVT();
  ev->done = 1;
  eq->ShiftEvent();                       // move on to next event
  eq->CommitAll(parent);
}
