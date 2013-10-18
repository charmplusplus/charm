/// Optimistic Synchronization Strategy No. 3: Time window
#include "pose.h"

void opt3::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  int iter = 0;

  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  // Prepare to execute an event
  ev = eq->currentPtr;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;
  
  if ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= lastGVT + timeLeash)) {
    POSE_TimeType fix_time = ev->timestamp;
    while (ev->timestamp == fix_time) {
      // do all events at the first available timestamp
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
    if (eq->currentPtr->timestamp > POSE_UnsetTS) {
      // execute next event if there is one
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-POSE_TimeMax);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

