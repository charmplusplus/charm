/// Optimistic Synchronization Strategy No. 3: Time window
#include "pose.h"

void opt3::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();

  // Prepare to execute an event
  ev = eq->currentPtr;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;
  
  if ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)) {
    POSE_TimeType fix_time = ev->timestamp;
    int iter = 0;
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
#ifdef POSE_STATS_ON
    if (iter > 0) localStats->Loop();
#endif  
    if (eq->currentPtr->timestamp >= 0) {
      // execute next event if there is one
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-POSE_TimeMax);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

