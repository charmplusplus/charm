/// Optimistic Synchronization Strategy No. 2
#include "pose.h"

void opt2::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();

  // Prepare to execute an event
  ev = eq->currentPtr;
  if ((ev->timestamp >= 0) && 
      ((POSE_endtime == POSE_UnsetTS) || (ev->timestamp <= POSE_endtime))){
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
    if (eq->currentPtr->timestamp >= 0) { // if more events, schedule the next
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-POSE_TimeMax);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

