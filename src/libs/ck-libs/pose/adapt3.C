/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
#define RANDOM_OBJECT 42
/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  static POSE_TimeType lastGVT = localPVT->getGVT();
  int iter=0;
  double critStart;
  rbFlag = 0;
 
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) {
    timeLeash = eq->RBevent->timestamp - lastGVT;
    Rollback();
  }
  if (!parent->cancels.IsEmpty()) CancelEvents();
 
  if (!rbFlag) timeLeash = (timeLeash + avgRBoffset)/2;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;
  // Prepare to execute an event
  ev = eq->currentPtr;
  while ((ev->timestamp > POSE_UnsetTS) &&
         (ev->timestamp <= lastGVT + timeLeash)) { // do events w/in timeLeash
#ifdef MEM_COARSE
    // note: first part of check below ensures we don't deadlock:
    //       can't advance gvt if we don't execute events with timestamp > gvt
    if (((eq->frontPtr->timestamp > lastGVT) ||
         (eq->frontPtr->timestamp < ev->prev->timestamp)) &&
        (eq->mem_usage > MAX_USAGE)) break;
#endif
    currentEvent = ev;
    ev->done = 2;
    specEventCount++;
    eventCount++;
#ifdef TRACE_DETAIL
    critStart = CmiWallTimer();  // trace timing
#endif
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
#ifdef TRACE_DETAIL
    traceUserBracketEvent(10, critStart, CmiWallTimer());
#endif
    ev->done = 1; // flag the event as executed
    eq->mem_usage++;
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
  if (!rbFlag && (ev->timestamp > POSE_UnsetTS)) 
    timeLeash = eq->largest - lastGVT;
  else if (!rbFlag && (timeLeash < avgTimeLeash)) timeLeash += LEASH_FLEX;
  // Uh oh!  Too much speculation going on!  Pull in the leash...
  if (specEventCount > (1.1*eventCount)) timeLeash = 1;
  rbFlag = 0;
  /*
  if (parent->thisIndex == RANDOM_OBJECT)
    CkPrintf("New leash=%d\n", timeLeash);
  */
}
 
