/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
/// Single forward execution step
void adapt3::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT(), offset;
  int iter=0;
  double critStart;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback();
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  if (rbFlag) { 
    if (timeLeash > avgRBoffset)
      timeLeash = avgRBoffset;
    else timeLeash = avgRBoffset/2;
  }
  if (specEventCount > (specTol*eventCount + eventCount)) {
    timeLeash = avgRBoffset;
  }
  else if ((specEventCount <= (specTol*eventCount + eventCount)) &&
	   (timeLeash < (POSE_TimeMax/2 -10))) {
    timeLeash += avgRBoffset;
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
  
  // Prepare to execute an event
  offset = lastGVT + timeLeash;
  if (offset < 0) offset = POSE_TimeMax;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && ((lastGVT+offset > POSE_endtime) ||
					(lastGVT+offset <= POSE_UnsetTS)))
    offset = POSE_endtime;

  ev = eq->currentPtr;
  //  CkPrintf("offset=%d timeLeash=%d avgRBoffset=%d specEventCount=%d eventCount=%d\n", offset, timeLeash, avgRBoffset, specEventCount, eventCount);
  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= offset)) { 
#ifdef MEM_COARSE
    // Check to see if we should hold off on forward execution to save on 
    // memory.
    // NOTE: to avoid deadlock, make sure we have executed something
    // beyond current GVT before worrying about memory usage
    if ((lastGVT < ev->prev->timestamp) &&
	(eq->mem_usage > pose_config.max_usage)) break;
#endif
    iter++;
    currentEvent = ev;
    ev->done = 2;
    specEventCount++;
    eventCount++;
#if !CMK_TRACE_DISABLED
    if(pose_config.trace)
      critStart = CmiWallTimer();  // trace timing
#endif
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
#if !CMK_TRACE_DISABLED
    if(pose_config.trace)
      traceUserBracketEvent(10, critStart, CmiWallTimer());
#endif
    ev->done = 1; // flag the event as executed
    eq->mem_usage++;
    eq->ShiftEvent(); // shift to next event
    ev = eq->currentPtr;
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (iter > 0) localStats->Loop();
  //if (iter > 5) CkPrintf("Executed %d events on this iteration; SE=%d E=%d\n", iter, specEventCount, eventCount);
#endif
}
 
