/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;
#ifdef POSE_STATS_ON
  int iter=0;
#endif

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) { // Cancel as much as possible
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(CAN_TIMER);      
#endif
    //CkPrintf("Trying to cancel events...\n");
    POSE_TimeType ct = eq->currentPtr->timestamp;  // store time of next event
    CancelEvents();
    // if cancellations of executed events occurred, adjust timeLeash
    if ((ct > -1) && (eq->currentPtr->timestamp < ct))
    timeLeash = eq->currentPtr->timestamp - lastGVT + 1;
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }
  if (RBevent) { // Rollback if necessary
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(RB_TIMER);      
#endif
    timeLeash = RBevent->timestamp - lastGVT;
    Rollback(); 
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }

  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > -1) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;

  // Prepare to execute an event
  ev = eq->currentPtr;
  // Avoid wasting this iteration by expanding speculative window to
  // include the earlier "half" of available work if there is any
  if ((eq->largest > -1) && (ev->timestamp > -1) && 
      (ev->timestamp > lastGVT + timeLeash))
    timeLeash += (eq->largest - ev->timestamp + 1)/2;
  while ((ev->timestamp > -1) && (ev->timestamp <= lastGVT + timeLeash)
	 && (iter < MAX_ITERATIONS)) { // do all events at & under timeLeash
    currentEvent = ev;
    ev->done = 2;
    //CkPrintf("About to do event "); ev->evID.dump(); CkPrintf("...\n");
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // flag the event as executed
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
#ifdef POSE_STATS_ON    
    iter++;
#endif
  }
#ifdef POSE_STATS_ON
  if (iter > 0) { 
    localStats->Loop();
    if (iter == MAX_ITERATIONS) CkPrintf("Touched MAX_ITERATIONS!\n");
  }
#endif  
  //if (ev->timestamp > -1) timeLeash = ev->timestamp - lastGVT + 1;
}

