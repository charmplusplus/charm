/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

/// Single forward execution step
void adapt2::Step()
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
    //timeLeash = RBevent->timestamp - lastGVT;
    //if (timeLeash < MIN_LEASH) timeLeash = MIN_LEASH;
    timeLeash = MIN_LEASH;
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

  //cpRate = (MAX_LEASH - timeLeash + MIN_LEASH)/5;
  int iter=0;
  while ((ev->timestamp > -1) && (ev->timestamp <= lastGVT + timeLeash)
	 && (iter < MAX_ITERATIONS)) {
    // do all events at under timeLeash
    iter++;
    currentEvent = ev;
    ev->done = 2;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
  }
#ifdef POSE_STATS_ON
  if (iter > 0) { 
    localStats->Loop();
    if (iter == MAX_ITERATIONS) CkPrintf("Touched MAX_ITERATIONS!\n");
  }
#endif  
  if (timeLeash < MAX_LEASH) timeLeash += LEASH_FLEX;
}

