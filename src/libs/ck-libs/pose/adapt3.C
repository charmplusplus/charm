/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;
  static int advances=0;
  int iter=0;

  lastGVT = localPVT->getGVT();
  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) { // Cancel as much as possible
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(CAN_TIMER);      
#endif
    //CkPrintf("Trying to cancel events...\n");
    //POSE_TimeType ct = eq->currentPtr->timestamp;  // store time of next event
    CancelEvents();
    // if cancellations of executed events occurred, adjust timeLeash
    /*
    if ((ct > -1) && (eq->currentPtr->timestamp < ct)) {
      timeLeash = eq->currentPtr->timestamp - lastGVT;
      advances = 1;
    }
    */
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }
  if (RBevent) { // Rollback if necessary
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(RB_TIMER);      
#endif
    //timeLeash = RBevent->timestamp - lastGVT;
    Rollback(); 
    //advances = 1;
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }

  // Prepare to execute an event
  ev = eq->currentPtr;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;

  while ((ev->timestamp > POSE_UnsetTS) && 
	 (ev->timestamp <= lastGVT + timeLeash) && 
	 (iter < MAX_ITERATIONS)) { // do all events at & under timeLeash
    currentEvent = ev;
    ev->done = 2;
    //CkPrintf("About to do event "); ev->evID.dump(); CkPrintf("...\n");
    specEventCount++;
    eventCount++;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // flag the event as executed
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
    iter++;
  }
  //if (ev->timestamp > POSE_UnsetTS) // work left undone
  //timeLeash = (eq->largest - ev->timestamp)/2 + ev->timestamp - lastGVT;
  // Calculate statistics for this run
  if (iter > 0) {
    avgTimeLeash = ((avgTimeLeash * stepCount) + timeLeash)/(stepCount+1);
    stepCount++;
#ifdef POSE_STATS_ON
    localStats->Loop();
#endif
  }
  avgEventsPerStep = specEventCount/stepCount;
  /*
  if (ev->timestamp > -1) {
    CkPrintf("%d BEFORE:timeLeash:%d nextWork:%d latestWork:%d iter:%d gvt:%d\n",
	     parent->thisIndex, timeLeash, ev->timestamp, eq->largest, iter,
	     lastGVT);
    CkPrintf(" avgTimeLeash:%d avgEventsPerStep:%d avgRBoffset:%d\n", 
	     avgTimeLeash, avgEventsPerStep, avgRBoffset);
  }
  */
  // Revise behavior for next run
  // immediate reaction
  if (rbFlag && (timeLeash > avgRBoffset))  // punish
    timeLeash -= (timeLeash-avgRBoffset)/2;
  if (!rbFlag && (iter < avgEventsPerStep) && (ev->timestamp > -1)
      && (timeLeash < (eq->largest-lastGVT))) //speculate
    timeLeash += (eq->largest - (timeLeash+lastGVT))/2;
  // reaction to past
  if ((avgRBoffset > POSE_UnsetTS) && (timeLeash > avgRBoffset)) // play safe
    timeLeash -= (timeLeash-avgRBoffset)/2;
  if (timeLeash < avgTimeLeash) // average out
    timeLeash += (avgTimeLeash-timeLeash)/2;
  else timeLeash -= (timeLeash-avgTimeLeash)/2;
  if (specEventCount > 1.5 * eventCount)
    timeLeash /= 2;
  // reaction to future
  if (timeLeash < (eq->largest - lastGVT)) // speculate
    timeLeash += (eq->largest - (timeLeash+lastGVT))/2;

  rbFlag = 0;
  /*
  if (ev->timestamp > -1) {
    CkPrintf("%d AFTER: timeLeash:%d nextWork:%d latestWork:%d iter:%d gvt:%d\n",
	     parent->thisIndex, timeLeash, ev->timestamp, eq->largest, iter,
	     lastGVT);
    CkPrintf(" avgTimeLeash:%d avgEventsPerStep:%d avgRBoffset:%d\n", 
	     avgTimeLeash, avgEventsPerStep, avgRBoffset);
  }
  */
}


