/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

/// Single forward execution step
void adapt2::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;
  int iter=0;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) {
    timeLeash = eq->RBevent->timestamp - lastGVT;
    if (timeLeash < MIN_LEASH) timeLeash = MIN_LEASH;
    Rollback(); 
  }
  if (!parent->cancels.IsEmpty()) CancelEvents();

  // Prepare to execute an event
  ev = eq->currentPtr;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;
  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= lastGVT + timeLeash)
	 && (iter < MAX_ITERATIONS)) {
    // do all events at under timeLeash
    idle = 0;
    iter++;
    currentEvent = ev;
    ev->done = 2;
    //CkPrintf("About to do event "); ev->evID.dump(); CkPrintf("...\n");
    //CkPrintf("POSE_DO\n");
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->mem_usage++;
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

