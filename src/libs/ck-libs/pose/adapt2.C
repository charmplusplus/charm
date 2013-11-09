/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

/// Single forward execution step
void adapt2::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  int iter=0;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  if (rbFlag) {
    timeLeash = eq->currentPtr->timestamp - lastGVT;
    if (timeLeash < pose_config.min_leash) timeLeash = pose_config.min_leash;
  }
  else if (timeLeash < pose_config.max_leash) timeLeash += pose_config.leash_flex;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;
  // Prepare to execute an event
  ev = eq->currentPtr;
  while ((ev->timestamp > POSE_UnsetTS) && 
	 (ev->timestamp <= lastGVT + timeLeash)) {
    // do all events at under timeLeash
    iter++;
    currentEvent = ev;
    ev->done = 2;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->mem_usage++;
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (iter > 0) localStats->Loop();
#endif
}

