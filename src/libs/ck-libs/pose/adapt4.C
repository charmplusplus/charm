/// Adaptive Synchronization Strategy No. 2
#include "pose.h"
 
/// Single forward execution step
void adapt4::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  static int itersAllowed=-1, iter=0, offset=-1, theMaxLeash=POSE_TimeMax/2, 
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
    timeLeash *= 2;
  }
  if (timeLeash > POSE_TimeMax-lastGVT) {
    timeLeash = theMaxLeash;
  }

  if (itersAllowed < 0) {
    itersAllowed = 10;
  }
  else {
    itersAllowed = (int)((double)specEventCount * specTol);
    itersAllowed -= specEventCount - eventCount;
    if (itersAllowed < 1) itersAllowed = 1;
  }
  
  // Prepare to execute an event
  offset = lastGVT + timeLeash;
  //  if (lastGVT == 15999) 
  //CkPrintf("itersAllowed=%d timeLeash=%d offset=%d nextTS=%d\n", itersAllowed, 
  //	     timeLeash, offset, eq->currentPtr->timestamp);
  //if (offset < 0) offset = POSE_TimeMax;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && ((offset > POSE_endtime) ||
					(offset <= POSE_UnsetTS)))
    offset = POSE_endtime;

  ev = eq->currentPtr;
  //CkPrintf("Object %d offset=%d timeLeash=%d itersAllowed=%d specEventCount=%d eventCount=%d\n", parent->thisIndex, offset, timeLeash, itersAllowed, specEventCount, eventCount);
  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= offset) &&
	 (itersAllowed > 0)) { 
#ifdef MEM_COARSE
    // note: first part of check below ensures we don't deadlock:
    //       can't advance gvt if we don't execute events with timestamp > gvt
    if ((ev->prev->timestamp > lastGVT) && (eq->mem_usage > objUsage)) {
      //      CkPrintf("Object %d waiting for new GVT: mem_usage=%d prev=%d gvt=%d\n", parent->thisIndex, eq->mem_usage, ev->prev->timestamp, lastGVT);
      break;
    }
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
  //if (iter > 5) CkPrintf("Executed %d events on this iteration; SE=%d E=%d\n", iter, specEventCount, eventCount);
#endif
}
 
