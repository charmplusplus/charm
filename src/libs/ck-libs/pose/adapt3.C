/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  int iter=0, offset;
  double critStart;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback();
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  if (rbFlag) { 
    timeLeash == avgRBoffset;
  }
  if (specEventCount > (specTol*eventCount)) {
    timeLeash--;
    if (timeLeash == 0) timeLeash = 1;
  }
  else if (specEventCount < (specTol*eventCount)) {
    timeLeash++;
  }
  /*
  if (rbFlag) timeLeash = 1;
  else if (eq->currentPtr->timestamp > POSE_UnsetTS)
    timeLeash = (timeLeash + (eq->largest - lastGVT))/2;
  else timeLeash = avgRBoffset;
    else if (specEventCount > (specTol*eventCount)) 
    timeLeash = 1;
    else if (specEventCount > (specTol*eventCount)-0.1)
    timeLeash = (timeLeash + avgRBoffset)/2;
  */
  
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT;
  // Prepare to execute an event
  offset = lastGVT + timeLeash;
  ev = eq->currentPtr;
  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= offset)) { 
#ifdef MEM_COARSE
    // note: first part of check below ensures we don't deadlock:
    //       can't advance gvt if we don't execute events with timestamp > gvt
    if (((eq->frontPtr->timestamp > lastGVT) ||
         (eq->frontPtr->timestamp < ev->prev->timestamp)) &&
        (eq->mem_usage > MAX_USAGE)) break;
#endif
    iter++;
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
  }
#ifdef POSE_STATS_ON
  if (iter > 0) localStats->Loop();
  //if (iter > 5) CkPrintf("Executed %d events on this iteration; SE=%d E=%d\n", iter, specEventCount, eventCount);
#endif
}
 
