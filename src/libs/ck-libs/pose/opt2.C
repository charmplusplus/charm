/// Optimistic Synchronization Strategy No. 2
#include "pose.h"

void opt2::Step()
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
  if ((ev->timestamp > POSE_UnsetTS) && 
      ((POSE_endtime == POSE_UnsetTS) || (ev->timestamp <= POSE_endtime))){
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
    if (eq->currentPtr->timestamp >= 0) { // if more events, schedule the next
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-POSE_TimeMax);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

