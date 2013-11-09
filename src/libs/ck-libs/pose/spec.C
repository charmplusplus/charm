/// Speculative Synchronization Strategy
#include "pose.h"

/// Single forward execution step
void spec::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  int iter = 0;

  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;

  // Prepare to execute an event
  ev = eq->currentPtr;
  while ((ev->timestamp > POSE_UnsetTS) && 
	 (ev->timestamp <= lastGVT + timeLeash)) {
    // do all events within the speculative window
    iter++;
    currentEvent = ev;
    ev->done = 2;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->mem_usage++;
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr; // reset ev
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (iter > 0) localStats->Loop();
#endif  
}

