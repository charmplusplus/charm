/// Basic Optimistic Synchronization Strategy
#include "pose.h"

/// Perform a single forward execution step
void opt::Step()
{
  Event *ev;
  static int lastGVT = -1;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) { // Cancel as much as possible
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(CAN_TIMER);      
#endif
    CancelEvents();
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }
  if (RBevent) { // Rollback if necessary
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(RB_TIMER);      
#endif
    Rollback(); 
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);      
#endif
  }

  // Execute an event
  ev = eq->currentPtr;
  if (ev->timestamp >= 0) {
    currentEvent = ev;
    ev->done = 2;
#ifdef POSE_STATS_ON
    localStats->Do();
    localStats->SwitchTimer(DO_TIMER);
#endif
    parent->DOs++;
    parent->ResolveFn(ev->fnIdx, ev->msg); // execute it
#ifdef POSE_STATS_ON
    localStats->SwitchTimer(SIM_TIMER);
#endif
    ev->done = 1; // complete the event execution
    eq->ShiftEvent(); // shift to next event
    if (eq->currentPtr->timestamp >= 0) { // if more events, schedule the next
      prioMsg *pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-INT_MAX);
      POSE_Objects[parent->thisIndex].Step(pm);
    }
  }
}

/// Rollback to predetermined RBevent
void opt::Rollback()
{
  Event *ev = eq->currentPtr->prev, *recoveryPoint;
  RBevent = eq->RecomputeRollbackTime();
  if (!RBevent) return; // no rollback necessary

  // find earliest event that must be undone
  recoveryPoint = RBevent;
  // skip forward over other stragglers
  while ((recoveryPoint != eq->back()) && (recoveryPoint->done == 0)) 
    recoveryPoint = recoveryPoint->next;
  CmiAssert(recoveryPoint != eq->back());

  // roll back over recovery point
#ifdef POSE_STATS_ON
  localStats->Rollback();
#endif
  while (ev != recoveryPoint) { // rollback, undoing along the way
    UndoEvent(ev); // undo the event
    ev = ev->prev;     
  }

  // ev is now at recovery point
  if (!recoveryPoint->cpData) { // no checkpoint, must recover state
    UndoEvent(recoveryPoint); // undo the recovery point
    RecoverState(recoveryPoint); // recover the state prior to target
  }
  else { // checkpoint available, simply undo
    targetEvent = recoveryPoint;
    UndoEvent(recoveryPoint); // undo the recovery point
  }

  eq->SetCurrentPtr(RBevent); // adjust currentPtr
  RBevent = targetEvent = NULL; // reset RBevent & targetEvent
}

/// Undo a single event, cancelling its spawned events
void opt::UndoEvent(Event *e)
{
  if (e->done == 1) {
    currentEvent = e;
    CancelSpawn(e); // cancel spawned events
#ifdef POSE_STATS_ON
    localStats->Undo();
#endif
    parent->UNDOs++;
    parent->ResolveFn(((e->fnIdx) * -1), e->msg); // execute the anti-method
    if (e->commitBfrLen > 0) free(e->commitBfr); // clean up buffered output
    e->commitBfr = NULL;
    e->commitErr = 0;
    e->done = e->commitBfrLen = 0;
    delete e->cpData;
    e->cpData = NULL;
  }
}

/// Cancel events in cancellation list that have arrived
void opt::CancelEvents() 
{
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
	while (!found && (ev->timestamp >= 0) && 
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
	  while (!found && (ev->timestamp >= 0) && 
		 (ev->timestamp >= it->timestamp)) {
	    if (ev->evID == it->evID)  found = 1; // found it
	    else ev = ev->prev;
	  }
	}
      }
      if (!found) { // "it" event has not arrived yet
	if (it == last) return; // seen all cancellations during this call
	it = parent->cancels.GetItem(); // try the next cancellation
      }
    }

    // something was found!
    if (ev && (ev->done == 0)) { // found it to be unexecuted; get rid of it
      if (ev == eq->currentPtr) eq->ShiftEvent(); // adjust currentPtr
      eq->DeleteEvent(ev); // delete the event
    }
    else if (ev) { // it's been executed, so rollback
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(RB_TIMER);
#endif
      recoveryPoint = ev; // ev is the target rollback point
      tmp = eq->currentPtr->prev;    
#ifdef POSE_STATS_ON
      localStats->Rollback();
#endif
      while (tmp != recoveryPoint) { // rollback, undoing along the way
	UndoEvent(tmp); // undo the event
	tmp = tmp->prev;
      }
      if (!recoveryPoint->cpData) { // no checkpoint, must recover state
	UndoEvent(recoveryPoint); // undo the recovery point
	RecoverState(recoveryPoint); // recover the state prior to target
      }
      else { // checkpoint available, simply undo
	targetEvent = recoveryPoint;
	UndoEvent(recoveryPoint); // undo the recovery point
      }
      eq->SetCurrentPtr(recoveryPoint->next); // adjust currentPtr
      eq->DeleteEvent(recoveryPoint); // delete the targetEvent
      targetEvent = NULL;
      // currentPtr may have unexecuted events in front of it
      while ((eq->currentPtr->prev->timestamp >= 0) 
	     && (eq->currentPtr->prev->done == 0))
	eq->currentPtr = eq->currentPtr->prev;
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(SIM_TIMER);
#endif
    }
    if (it == last) {
      parent->cancels.RemoveItem(it); // Clean up
      if (RBevent && (RBevent->timestamp > eq->currentPtr->timestamp))
	RBevent = NULL;
      return;
    }
    else parent->cancels.RemoveItem(it); // Clean up
  } // end outer while which loops through entire cancellations list
  if (RBevent && (RBevent->timestamp > eq->currentPtr->timestamp))
    RBevent = NULL;
}

/// Recover checkpointed state prior to ev
void opt::RecoverState(Event *recoveryPoint)
{
  Event *ev;
  CmiAssert(!recoveryPoint->cpData); // ERROR: recoveryPoint has checkpoint
  CpvAccess(stateRecovery) = 1; // FE behavior changed to state recovery only
  ev = recoveryPoint->prev;
  while ((ev != eq->front()) && (!ev->cpData)) { // search for checkpoint
    if (ev->commitBfrLen > 0)
      free(ev->commitBfr);
    ev->commitBfr = NULL;
    ev->commitBfrLen = 0;
    ev = ev->prev;
  }
  CmiAssert(ev != eq->front()); // ERROR: no checkpoint

  // restore state from ev->cpData
  currentEvent = targetEvent = ev;
  parent->ResolveFn(((ev->fnIdx) * -1), ev->msg);
  if (ev->commitBfrLen > 0)
    free(ev->commitBfr);
  ev->commitBfr = NULL;
  ev->commitBfrLen = 0;
  delete ev->cpData;
  ev->cpData = NULL;
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
