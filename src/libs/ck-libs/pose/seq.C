// File: seq.C
// Module for sequential simulation strategy class
#include "pose.h"

void seq::Step()
{
  Event *ev;
  // Prepare to execute an event
  ev = eq->currentPtr;
  if (ev->timestamp >= 0) { 
    currentEvent = ev;
    ev->done = 2;
    parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
    ev->done = 1;
    eq->ShiftEvent();                       // move on to next event
    eq->CommitAll(parent);
  }
}
