/// Adaptive Synchronization Strategy No. 2
#include "pose.h"

#define RANDOM_OBJECT 42
/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = POSE_UnsetTS;
  static int advances=0;
  int iter=0;

  rbFlag = 0;
  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) {
    timeLeash = eq->RBevent->timestamp - lastGVT;
    Rollback(); 
  }
  if (!parent->cancels.IsEmpty()) CancelEvents();

  if (!rbFlag) timeLeash = (timeLeash + avgRBoffset)/2;
  // Prepare to execute an event
  ev = eq->currentPtr;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;
  while ((ev->timestamp > POSE_UnsetTS) && 
	 (ev->timestamp <= lastGVT + timeLeash)) { // do events w/in timeLeash
    currentEvent = ev;
    ev->done = 2;
    specEventCount++;
    eventCount++;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // flag the event as executed
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
    iter++;
  }
  // Calculate statistics for this run
  if (iter > 0) {
    avgTimeLeash = ((avgTimeLeash * stepCount) + timeLeash)/(stepCount+1);
    stepCount++;
#ifdef POSE_STATS_ON
    localStats->Loop();
#endif
  }
  if (stepCount > 0)  avgEventsPerStep = specEventCount/stepCount;
  /*
  if (parent->thisIndex == RANDOM_OBJECT) {
    CkPrintf("%d STATS: leash:%d work:%d max:%d gvt:%d\n",
	     parent->thisIndex, timeLeash, ev->timestamp, eq->largest,lastGVT);
    CkPrintf(" avgLeash:%d RB:%d:%d specEvents=%d events=%d\n", 
	     avgTimeLeash, avgRBoffset, rbFlag, specEventCount, eventCount);
  }
  */
  // Revise behavior for next run
  if (!rbFlag && (ev->timestamp > -1)) timeLeash = eq->largest - lastGVT;
  else if (!rbFlag && (timeLeash < avgTimeLeash)) timeLeash += LEASH_FLEX;
  // Uh oh!  Too much speculation going on!  Pull in the leash...
  if (specEventCount > (1.1*eventCount)) timeLeash = 1;
  /*
  if (parent->thisIndex == RANDOM_OBJECT)
    CkPrintf("New leash=%d\n", timeLeash);
  */
  rbFlag = 0;
}


