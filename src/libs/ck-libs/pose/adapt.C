/// Adaptive Synchronization Strategy
#include "pose.h"

/// Single forward execution step
void adapt::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  int iter=0;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  if (rbFlag) timeLeash = pose_config.min_leash;
  else if (timeLeash < pose_config.max_leash) timeLeash += pose_config.leash_flex; //expand spec window
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
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (iter > 0) localStats->Loop();
#endif  
}

