// File: opt3.C
#include "pose.h"

opt3::opt3() { timeLeash = SPEC_WINDOW; STRAT_T = OPT3_T; }

// Single forward execution step
void opt3::Step()
{
  Event *ev;
  static int lastGVT = 0;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) {             // Cancel as much as possible
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
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > -1) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;
  
  if ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)) {
    int fix_time = ev->timestamp;
    while (ev->timestamp == fix_time) {
      // do all events at the first available timestamp
      currentEvent = ev;
      ev->done = 2;
#ifdef POSE_STATS_ON
      localStats->Do();
      localStats->SwitchTimer(DO_TIMER);
#endif
      parent->DOs++;
      parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(SIM_TIMER);
#endif
      ev->done = 1;                           // complete the event execution
      eq->ShiftEvent();                       // shift to next event
      ev = eq->currentPtr;                    // reset ev
    }
    if (eq->currentPtr->timestamp >= 0)
      parent->Step();             // execute next event if there is one
  }
}

