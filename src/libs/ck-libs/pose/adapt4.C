/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
/// Single forward execution step
void adapt4::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  static int itersAllowed=-1, iter=0, offset=-1, theMaxLeash=10, 
    objUsage = MAX_USAGE * STORE_RATE;
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
  else if (timeLeash < theMaxLeash) {
    timeLeash++;
  }
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
#endif
}
 
