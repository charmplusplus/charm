/// Adaptive Synchronization Strategy
#include "pose.h"

/// Single forward execution step
void adapt::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = localPVT->getGVT();
  int iter=0;

  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) {
    timeLeash = MIN_LEASH; // shrink speculative window
    Rollback(); 
  }
  if (!parent->cancels.IsEmpty()) CancelEvents();

  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;

  // Prepare to execute an event
  ev = eq->currentPtr;
  while ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)) {
    // do all events within speculative window
    currentEvent = ev;
    ev->done = 2;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->mem_usage++;
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
    iter++;
  }
#ifdef POSE_STATS_ON
  if (iter > 0) localStats->Loop();
#endif  
  if (timeLeash < MAX_LEASH) timeLeash += LEASH_FLEX; // expand spec window
}

