/// Basic Optimistic Synchronization Strategy
#include "pose.h"

/// Perform a single forward execution step
void opt::Step()
{
  Event *ev;
  POSE_TimeType lastGVT = localPVT->getGVT();

  if (!parent->cancels.IsEmpty()) CancelUnexecutedEvents();
  if (eq->RBevent) Rollback(); 
  if (!parent->cancels.IsEmpty()) CancelEvents();
  parent->Status();

  // Execute an event
  ev = eq->currentPtr;
  if (ev->timestamp > POSE_UnsetTS) {
    idle = 0;
    currentEvent = ev;
    ev->done = 2;
    specEventCount++;
    eventCount++;
    stepCount++;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
    ev->done = 1; // complete the event execution
    eq->mem_usage++;
    eq->ShiftEvent(); // shift to next event
    if (eq->currentPtr->timestamp > POSE_UnsetTS) { // if more events, schedule the next
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-POSE_TimeMax);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->Loop();
#endif  
  }
}

/// Rollback to predetermined RBevent
void opt::Rollback()
{
#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart = CmiWallTimer();  // trace timing
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(RB_TIMER);      
#endif
  Event *ev = eq->currentPtr->prev, *recoveryPoint;
  // find earliest event that must be undone
  recoveryPoint = eq->RBevent;
  // skip forward over other stragglers
  while ((recoveryPoint != eq->back()) && (recoveryPoint->done == 0) 
	 && (recoveryPoint != eq->currentPtr)) 
    recoveryPoint = recoveryPoint->next;
  if (recoveryPoint == eq->currentPtr) {
    eq->SetCurrentPtr(eq->RBevent);
    eq->RBevent = NULL;
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
    if(pose_config.trace)
      traceUserBracketEvent(40, critStart, CmiWallTimer());
#endif
    return;
  }
  CmiAssert(recoveryPoint != eq->back());
  CmiAssert(eq->RBevent->prev->next == eq->RBevent);
  CmiAssert(eq->RBevent->next->prev == eq->RBevent);

  rbCount++;
  rbFlag = 1;
  // roll back over recovery point
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->Rollback();
#endif
  while (ev != recoveryPoint) { // rollback, undoing along the way
    UndoEvent(ev); // undo the event
    ev = ev->prev;     
  }

  // ev is now at recovery point
  if (userObj->usesAntimethods()) {
    targetEvent = recoveryPoint;
    UndoEvent(recoveryPoint); // undo the recovery point
  }
  else {
#ifdef MEM_TEMPORAL    
    if (!recoveryPoint->serialCPdata) { // no checkpoint, must recover state
#else
    if (!recoveryPoint->cpData) { // no checkpoint, must recover state
#endif
      UndoEvent(recoveryPoint); // undo the recovery point
      RecoverState(recoveryPoint); // recover the state prior to target
    }
    else { // checkpoint available, simply undo
      targetEvent = recoveryPoint;
      UndoEvent(recoveryPoint); // undo the recovery point
    }
  }

  eq->SetCurrentPtr(eq->RBevent); // adjust currentPtr
  avgRBoffset = 
    (avgRBoffset*(rbCount-1)+(eq->currentPtr->timestamp-localPVT->getGVT()))/rbCount;
  eq->FindLargest();
  eq->RBevent = targetEvent = NULL; // reset RBevent & targetEvent
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.trace)
    traceUserBracketEvent(40, critStart, CmiWallTimer());
#endif
}

/// Undo a single event, cancelling its spawned events
void opt::UndoEvent(Event *e)
{
  if (e->done == 1) {
    //CkPrintf("[%d] undo ", CkMyPe()); e->evID.dump(); CkPrintf("...\n");
    eq->eventCount++;
    currentEvent = e;
    CancelSpawn(e); // cancel spawned events
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->Undo();
#endif
    parent->UNDOs++;
    localPVT->decEventCount();
    eventCount--;
    //CkPrintf("POSE_UNDO\n");
    parent->ResolveFn(((e->fnIdx) * -1), e->msg); // execute the anti-method
    parent->basicStats[1]++;
    if (e->commitBfrLen > 0) free(e->commitBfr); // clean up buffered output
    e->commitBfr = NULL;
    e->commitErr = 0;
    e->done = e->commitBfrLen = 0;
#ifdef MEM_TEMPORAL
    if (e->serialCPdata) {
      userObj->localTimePool->tmp_free(e->timestamp, e->serialCPdata);
      e->serialCPdata = NULL; 
      e->serialCPdataSz = 0;
    }
#else
    delete e->cpData;
    e->cpData = NULL;
#endif
    eq->mem_usage--;
  }
}

/// Cancel events in cancellation list that have arrived
void opt::CancelEvents() 
{
#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart = CmiWallTimer();  // trace timing
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(CAN_TIMER);      
#endif
  Event *ev, *tmp, *recoveryPoint;
  int found;
  CancelNode *it=NULL, *last=NULL;

  last = parent->cancels.GetItem(); // make note of last item to examine
  while (!parent->cancels.IsEmpty()) { // loop through all cancellations
    it = parent->cancels.GetItem();
    found = 0; // init the found flag to not found (0)
    // search cancellations list for a cancellation that has a corresponding
    // event in the event queue
    while (!found) {  // loop until one is found, or exit fn if all examined
      //CkPrintf("Trying to cancel "); it->evID.dump(); CkPrintf(" at %d...\n", it->timestamp);
      ev = eq->currentPtr;               // set search start point
      if (ev == eq->back()) ev = ev->prev;
      if (ev->timestamp <= it->timestamp) {
	// search forward for 'it' from currentPtr to backPtr
	while (!found && (ev->timestamp > POSE_UnsetTS) && 
	       (ev->timestamp <= it->timestamp)) {
	  if (ev->evID == it->evID) found = 1; // found it
	  else ev = ev->next;
	}
	if (!found) { // not in linked list; check the heap
	  found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	  if (found) ev = NULL; // make ev NULL so we know it was deleted
	}
      }
      if (!found) { 
	ev = eq->currentPtr; // set search start point
	if (ev == eq->back()) ev = ev->prev;
	if (ev->timestamp >= it->timestamp) { // search backward
	  while (!found && (ev->timestamp > POSE_UnsetTS) && 
		 (ev->timestamp >= it->timestamp)) {
	    if (ev->evID == it->evID)  found = 1; // found it
	    else ev = ev->prev;
	  }
	}
      }
      if (!found) { // "it" event has not arrived yet
	if (it == last) {
#if !CMK_TRACE_DISABLED
	  if(pose_config.stats)
	    localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
	  if(pose_config.trace)
	    traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
	  return;  // seen all cancellations during this call
	}
	it = parent->cancels.GetItem(); // try the next cancellation
      }
    }

    // something was found!
    if (ev && (ev->done == 0)) { // found it to be unexecuted; get rid of it
      if (ev == eq->currentPtr) eq->ShiftEvent(); // adjust currentPtr
      //CkPrintf("Cancelled event "); ev->evID.dump(); CkPrintf(" deleted!\n");
      //CkPrintf("POSE_CANCEL_DELETE\n");
      eq->DeleteEvent(ev); // delete the event
    }
    else if (ev) { // it's been executed, so rollback
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->SwitchTimer(RB_TIMER);
#endif
      recoveryPoint = ev; // ev is the target rollback point
      tmp = eq->currentPtr->prev;    
#if !CMK_TRACE_DISABLED
      if(pose_config.stats)
	localStats->Rollback();
#endif
      while (tmp != recoveryPoint) { // rollback, undoing along the way
	UndoEvent(tmp); // undo the event
	tmp = tmp->prev;
      }
      rbFlag = 1;
      if (userObj->usesAntimethods()) {
	targetEvent = recoveryPoint;
	UndoEvent(recoveryPoint); // undo the recovery point
      }
      else {
#ifdef MEM_TEMPORAL
	if (!recoveryPoint->serialCPdata) { // no checkpoint, must recover state
#else
	if (!recoveryPoint->cpData) { // no checkpoint, must recover state
#endif
	  UndoEvent(recoveryPoint); // undo the recovery point
	  RecoverState(recoveryPoint); // recover the state prior to target
	}
	else { // checkpoint available, simply undo
	  targetEvent = recoveryPoint;
	  UndoEvent(recoveryPoint); // undo the recovery point
	}
      }
      eq->SetCurrentPtr(recoveryPoint->next); // adjust currentPtr 
      //CkPrintf("Cancelled event "); ev->evID.dump(); CkPrintf(" deleted!\n");
      //CkPrintf("POSE_CANCEL_DELETE_UNDO\n");
      eq->DeleteEvent(recoveryPoint); // delete the targetEvent
      targetEvent = NULL;
      // currentPtr may have unexecuted events in front of it
      while ((eq->currentPtr->prev->timestamp > POSE_UnsetTS) 
	     && (eq->currentPtr->prev->done == 0))
	eq->currentPtr = eq->currentPtr->prev;
#if !CMK_TRACE_DISABLED
      if(pose_config.stats)
	localStats->SwitchTimer(CAN_TIMER);
#endif
    }
    if (it == last) {
      parent->cancels.RemoveItem(it); // Clean up
#if !CMK_TRACE_DISABLED
      if(pose_config.stats)
	localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
      if(pose_config.trace)
	traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
      return;
    }
    else parent->cancels.RemoveItem(it); // Clean up
  } // end outer while which loops through entire cancellations list
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.trace)
    traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
}

/// Cancel events in cancellation list that have arrived
void opt::CancelUnexecutedEvents() 
{
#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart = CmiWallTimer();  // trace timing
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(CAN_TIMER);      
#endif
  Event *ev, *tmp, *recoveryPoint;
  int found;
  CancelNode *it=NULL, *last=NULL;

  last = parent->cancels.GetItem(); // make note of last item to examine
  while (!parent->cancels.IsEmpty()) { // loop through all cancellations
    it = parent->cancels.GetItem();
    found = 0; // init the found flag to not found (0)
    // search cancellations list for a cancellation that has a corresponding
    // event in the event queue
    while (!found) {  // loop until one is found, or exit fn if all examined
      ev = eq->currentPtr;               // set search start point
      if (ev == eq->back()) ev = ev->prev;
      if (ev->timestamp <= it->timestamp) {
	// search forward for 'it' from currentPtr to backPtr
	while (!found && (ev->timestamp > POSE_UnsetTS) && 
	       (ev->timestamp <= it->timestamp)) {
	  if (ev->evID == it->evID) found = 1; // found it
	  else ev = ev->next;
	}
	if (!found) { // not in linked list; check the heap
	  found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	  if (found) ev = NULL; // make ev NULL so we know it was deleted
	}
      }
      if (!found) { 
	ev = eq->currentPtr; // set search start point
	if (ev == eq->back()) ev = ev->prev;
	if (ev->timestamp >= it->timestamp) { // search backward
	  while (!found && (ev->timestamp > POSE_UnsetTS) && 
		 (ev->timestamp >= it->timestamp)) {
	    if (ev->evID == it->evID)  found = 1; // found it
	    else ev = ev->prev;
	  }
	}
      }
      if (!found) { // "it" event has not arrived yet
	if (it == last) {
#if !CMK_TRACE_DISABLED
	  if(pose_config.stats)
	    localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
	  if(pose_config.trace)
	    traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
	  return;  // seen all cancellations during this call
	}
	it = parent->cancels.GetItem(); // try the next cancellation
      }
    }

    if (ev && (ev->done == 0)) { // found it to be unexecuted; get rid of it
      if (ev == eq->currentPtr) eq->ShiftEvent(); // adjust currentPtr
      eq->DeleteEvent(ev); // delete the event
      if (it == last) {
	parent->cancels.RemoveItem(it); // Clean up
#if !CMK_TRACE_DISABLED
	if(pose_config.stats)
	  localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
	if(pose_config.trace)
	  traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
	return;
      }
      else parent->cancels.RemoveItem(it); // Clean up
    }
    else if (ev && (ev->done == 1)) { 
      if (it == last) {
#if !CMK_TRACE_DISABLED
	if(pose_config.stats)
	  localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
	if(pose_config.trace)
	  traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
	return; 
      }
    }
    else if (!ev) {
      if (it == last) {
	parent->cancels.RemoveItem(it); // Clean up
#if !CMK_TRACE_DISABLED
	if(pose_config.stats)
	  localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
	if(pose_config.trace)
	  traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
	return;
      }
      else parent->cancels.RemoveItem(it); // Clean up
    }
  } // end outer while which loops through entire cancellations list
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);      
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.trace)
    traceUserBracketEvent(20, critStart, CmiWallTimer());
#endif
}

/// Recover checkpointed state prior to ev
void opt::RecoverState(Event *recoveryPoint)
{
  Event *ev;
#ifdef MEM_TEMPORAL
  CmiAssert(!recoveryPoint->serialCPdata); // ERROR: recoveryPoint has checkpoint
#else
  CmiAssert(!recoveryPoint->cpData); // ERROR: recoveryPoint has checkpoint
#endif
  CpvAccess(stateRecovery) = 1; // FE behavior changed to state recovery only
  ev = recoveryPoint->prev;
#ifdef MEM_TEMPORAL
  while ((ev != eq->front()) && (!ev->serialCPdata)) { // search for checkpoint
#else
  while ((ev != eq->front()) && (!ev->cpData)) { // search for checkpoint
#endif
    if (ev->commitBfrLen > 0)
      free(ev->commitBfr);
    ev->commitBfr = NULL;
    ev->commitBfrLen = 0;
    ev = ev->prev;
  }
  CmiAssert(ev != eq->front()); // ERROR: no checkpoint

  // restore state from ev->cpData or ev->serialCPdata
  currentEvent = targetEvent = ev;
  parent->ResolveFn(((ev->fnIdx) * -1), ev->msg);
  parent->basicStats[1]++;
  if (ev->commitBfrLen > 0)
    free(ev->commitBfr);
  ev->commitBfr = NULL;
  ev->commitBfrLen = 0;
#ifdef MEM_TEMPORAL
  if (ev->serialCPdata) {
    userObj->localTimePool->tmp_free(ev->timestamp, ev->serialCPdata);
    ev->serialCPdata = NULL; 
    ev->serialCPdataSz = 0;
  }
#else
  delete ev->cpData;
  ev->cpData = NULL;
#endif
  targetEvent = NULL;

  // execute forward to recoveryPoint
  while (ev != recoveryPoint) {
    if (ev->done == 1) {
      currentEvent = ev;
      parent->ResolveFn(ev->fnIdx, ev->msg);
    }
    ev = ev->next;
  }
  CpvAccess(stateRecovery) = 0; // return forward execution behavior to normal
}
