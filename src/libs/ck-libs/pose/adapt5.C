/// Adaptive Synchronization Strategy No. 5
#include "pose.h"

/// Single forward execution step
void adapt5::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();
  POSE_TimeType maxTimeLeash, offset;
  double critStart;

  rbFlag = 0;
  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback();
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

/*
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
*/


/*
  int i = timeLeash >> 6;
  if (i > 1) {
    if (rbFlag) {
      timeLeash -= i;
    } else {
      timeLeash += i;
    }
  } else {
    if (rbFlag) {
      timeLeash--;
    } else {
      timeLeash++;
    }
  }

  if (timeLeash <= 0) {
    timeLeash = 10;
  }
*/
/*
  if (rbFlag) {
    timeLeash = timeLeash >> 1;
  } else {
    timeLeash++;
  }
*/

// other possibilities:
// -run for a number of events with a fixed timeleash, then analyze
// and adjust
// -simply use recentAvgEventSparsity * avgEventsPerRB
// -add code that looks at the number of RBs this poser is responsible
// for and don't punish it if it doesn't have a lot

  ev = eq->currentPtr;

  if (parent->basicStats[1] > 0) {
    avgEventsPerRB = (int)(parent->basicStats[0] / parent->basicStats[1]);
    if (avgEventsPerRB < 1) {
      avgEventsPerRB = 1;
    }
  }

  // ======== attempt 1 ========
#if ALGORITHM_TO_USE == 1
  if (rbFlag) {
    recentAvgRBLeashCount++;
    recentTotalRBLeash += timeLeash;
    // initial rollback calculation to quickly set recentAvgRBLeash to a reasonable value
    if (initialAvgRBLeashCalc) {
      recentAvgRBLeash = avgRBoffset;
      recentTotalRBLeash = 0;
      recentAvgRBLeashCount = 0;
      initialAvgRBLeashCalc = false;
    }
    // calculate the recent average timeleash when rollbacks occur
    if (recentAvgRBLeashCount >= AVG_LEASH_CALC_PERIOD) {
      recentAvgRBLeash = recentTotalRBLeash / recentAvgRBLeashCount;
      recentTotalRBLeash = 0;
      recentAvgRBLeashCount = 0;
    }
    if (timeLeash > recentAvgRBLeash) {
      timeLeash = recentAvgRBLeash;
    } else {
      timeLeash = recentAvgRBLeash / 2;
    }
  } else {
    timeLeash += recentAvgEventSparsity;
  }

  if (avgRBsPerGVTIter > MAX_RB_PER_GVT_ITER) {
    maxTimeLeash = recentAvgRBLeash / 2;
  } else {
    maxTimeLeash = (POSE_TimeType)MAX_LEASH_MULTIPLIER * (POSE_TimeType)recentAvgEventSparsity * (POSE_TimeType)avgEventsPerRB;
  }

  if (maxTimeLeash > 50000) {
    maxTimeLeash = 50000;
  }
  if (timeLeash > maxTimeLeash) {
    timeLeash = maxTimeLeash;
  }
  if (timeLeash < 1) {
    timeLeash = 1;
  }
#endif

  // ======== attempt 2 ========
#if ALGORITHM_TO_USE == 2
  timeLeash = recentAvgEventSparsity * avgEventsPerRB;

  if (timeLeash > 50000) {
    timeLeash = 50000;
  }

  if (timeLeash < 1) {
    timeLeash = 1;
  }
#endif

  // ======== attempt 3 ========
#if ALGORITHM_TO_USE == 3
  timeLeash += recentAvgEventSparsity;

  if (avgRBsPerGVTIter > MAX_RB_PER_GVT_ITER) {
    maxTimeLeash = ((POSE_TimeType)MAX_LEASH_MULTIPLIER * (POSE_TimeType)recentAvgEventSparsity * (POSE_TimeType)avgEventsPerRB) / (4 * (POSE_TimeType)avgRBsPerGVTIter);
  } else {
    maxTimeLeash = 1000;
  }  

  if (maxTimeLeash > 50000) {
    maxTimeLeash = 50000;
  }
  if (timeLeash > maxTimeLeash) {
    timeLeash = maxTimeLeash;
  }
  if (timeLeash < 1) {
    timeLeash = 1;
  }
#endif

  // ======== attempt 4 ========
#if ALGORITHM_TO_USE == 4
  if (timeLeash > 10000) {
    timeLeash = 10000;
  }
  if (timeLeash < 1) {
    timeLeash = 1;
  }
#endif


/*
  if (rbFlag) {
    GVT *localGVT = (GVT *)CkLocalBranch(TheGVT);
    int numRollbacks = parent->basicStats[1];
    int numGVTIters = localGVT->gvtIterationCount;
    if (userObj->myHandle == 32) {
      CkPrintf("*** ROLLBACK: numRollbacks=%d numGVTIters=%d\n", numRollbacks, numGVTIters);
    }
    if ((numGVTIters > 0) && (((96 * numRollbacks) / numGVTIters) > 2)) {
      timeLeash = avgRBoffset;
    } else {
      timeLeash++;
    }
  } else {
    timeLeash++;
  }

  if (timeLeash > (avgRBoffset << 1)) {
    timeLeash = avgRBoffset << 1;
  }
*/


  // can also just hard-code the time leash
//  timeLeash = 1000;


  //  if (stepCalls == 0) {
  //    timeLeashTotal = 0LL;
  //  }

  //  if (timeLeash < 1000000) {
  //    stepCalls++;
  //    timeLeashTotal += timeLeash;
  //  }

/*
  if (itersAllowed < 1) {
    itersAllowed = 1;
  }
  else {
    itersAllowed = (int)((double)specEventCount * specTol);
    itersAllowed -= specEventCount - eventCount;
    if (itersAllowed < 1) itersAllowed = 1;
  }
*/


  // Prepare to execute an event
  offset = lastGVT + timeLeash;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > POSE_UnsetTS) && ((offset > POSE_endtime) ||
					(offset <= POSE_UnsetTS)))
    offset = POSE_endtime;

  while ((ev->timestamp > POSE_UnsetTS) && (ev->timestamp <= offset) ){
#ifdef MEM_COARSE
    // Check to see if we should hold off on forward execution to save on 
    // memory.
    // NOTE: to avoid deadlock, make sure we have executed something
    // beyond current GVT before worrying about memory usage
    if (((ev->timestamp > lastGVT) || (userObj->OVT() > lastGVT))
	&& (eq->mem_usage > objUsage)) { // don't deadlock
      break;
    }
#endif

    iter++;
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
