// File: adapt2.C
#include "pose.h"

adapt2::adapt2() { 
  timeLeash = MIN_LEASH; 
  eventLeash = MAX_EVENTS; 
  STRAT_T = ADAPT2_T; 
}

// Single forward execution step
void adapt2::Step()
{
  Event *ev;
  static int lastGVT = 0;
  static int evCount = 0;

  if (localPVT->getGVT() > lastGVT) {
    lastGVT = localPVT->getGVT();
    evCount = 0;
  } 
  else if (eq->currentPtr->timestamp == lastGVT) {
    evCount = 0;
  }
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
    //    CkPrintf("<ROLLBACK of %d:%d @ time %d (L_t:%d, L_e:%d) OVT=%d GVT=%d\n", RBevent->evID.id, RBevent->evID.pe, RBevent->timestamp, timeLeash, eventLeash, userObj->OVT(), lastGVT);
    timeLeash = MIN_LEASH;
    eventLeash = MIN_EVENTS;
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

  while ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)
	 && (evCount < eventLeash)) {
    // do all events at under timeLeash
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
    ev = eq->currentPtr;
    evCount++;
  }
  if (timeLeash < MAX_LEASH) timeLeash++;
  if (eventLeash < MAX_EVENTS) eventLeash++;
}

