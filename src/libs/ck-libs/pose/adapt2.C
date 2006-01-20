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
    if (timeLeash < MIN_LEASH) timeLeash = MIN_LEASH;
  }
  else if (timeLeash < MAX_LEASH) timeLeash += LEASH_FLEX;
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
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    if (iter > 0) localStats->Loop();
#endif
}

