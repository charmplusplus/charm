/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
/// Single forward execution step
void adapt4::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  POSE_TimeType offset=POSE_UnsetTS, theMaxLeash=POSE_TimeMax/2;
  double critStart;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback();
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  if (rbFlag) { // adjust leash according to rollback
    timeLeash = avgRBoffset;
  }
  else if (timeLeash < theMaxLeash) { // adjust according to state
    if (eq->currentPtr->timestamp > POSE_UnsetTS) { // adjust to next event
      if (eq->currentPtr->timestamp - lastGVT > timeLeash)
	timeLeash == eq->currentPtr->timestamp - lastGVT;
      // else leave it alone
    }
    // no next event; leave it alone
  }
  // Put leash back into reasonable bounds
  if (timeLeash > theMaxLeash) {
    timeLeash = theMaxLeash;
  }
  else if (timeLeash < 0) timeLeash = 0;

  if (itersAllowed < 1) {
    itersAllowed = 1;
  }
  else {
    itersAllowed = (int)((double)specEventCount * specTol);
    itersAllowed -= specEventCount - eventCount;
    if (itersAllowed < 1) itersAllowed = 1;
  }
  
  // Prepare to execute an event
  offset = lastGVT + timeLeash;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && ((offset > POSE_endtime) ||
					(offset <= POSE_UnsetTS)))
    offset = POSE_endtime;

  ev = eq->currentPtr;
  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= offset) &&
	 (itersAllowed > 0)) { 
#ifdef MEM_COARSE
    // Check to see if we should hold off on forward execution to save on 
    // memory.
    // NOTE: to avoid deadlock, make sure we have executed something
    // beyond current GVT before worrying about memory usage
    if (((ev->timestamp > lastGVT) || (userObj->OVT() > lastGVT))
	&& (eq->mem_usage > objUsage)) // don't deadlock
      break;
#endif
    iter++;
    itersAllowed--;
    currentEvent = ev;
    ev->done = 2;
    localPVT->incSpecEventCount();
    localPVT->incEventCount();
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
#endif
}
 
