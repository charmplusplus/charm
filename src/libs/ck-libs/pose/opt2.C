/// Optimistic Synchronization Strategy No. 2
#include "pose.h"

void opt2::Step()
{
  Event *ev;
  static int lastGVT = -1;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) { // Cancel as much as possible
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(CAN_TIMER);      
#endif
    CancelEvents();
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }
  if (RBevent) { // Rollback if necessary
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(RB_TIMER);      
#endif
    Rollback(); 
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }

  // Prepare to execute an event
  ev = eq->currentPtr;
  if ((ev->timestamp >= 0) && 
      ((POSE_endtime == -1) || (ev->timestamp <= POSE_endtime))){
    int fix_time = ev->timestamp;
    while (ev->timestamp == fix_time) {
      // do all events at the first available timestamp
      currentEvent = ev;
      ev->done = 2;
      parent->DOs++;
      parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(SIM_TIMER);
#endif
      ev->done = 1; // complete the event execution
      eq->ShiftEvent(); // shift to next event
      ev = eq->currentPtr; // reset ev
    }
    if (eq->currentPtr->timestamp >= 0) { // if more events, schedule the next
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-INT_MAX);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

