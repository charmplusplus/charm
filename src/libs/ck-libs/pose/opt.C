// File: opt.C
// Module for optimistic simulation strategy class
// Last Modified: 09.12.01 by Terry L. Wilmarth

#include "pose.h"

opt::opt() { timeLeash = SPEC_WINDOW; }

// Single forward execution step
void opt::Step()
{
  Event *ev;
  static int lastGVT = 0;

  lastGVT = localPVT->getGVT();
  if (!parent->cancels.IsEmpty()) {             // Cancel as much as possible
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
  // Prepare to execute an event
  ev = eq->currentPtr;
  if (ev->timestamp < 0) return;
  // Shorten the leash as we near POSE_endtime
  if ((POSE_endtime > -1) && (lastGVT + timeLeash > POSE_endtime))
    timeLeash = POSE_endtime - lastGVT + 1;

  // This code is currently operating under BEST FIRST mode
  if ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)) {
    int fix_timestamp = ev->timestamp;
    while (ev->timestamp == fix_timestamp) {
      // do all events with fix_timestamp
      currentEvent = ev;
      ev->done = 2;
#ifdef POSE_STATS_ON
      localStats->Do();
      localStats->SwitchTimer(DO_TIMER);
#endif
      parent->DOs++;
      parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(SIM_TIMER);
#endif
      ev->done = 1;                           // complete the event execution
      eq->ShiftEvent();                       // shift to next event
      ev = eq->currentPtr;
    }
  }
  if ((ev->timestamp >= 0) && (ev->timestamp <= lastGVT + timeLeash)) {
    prioMsg *pm = new prioMsg;
    pm->setPriority(eq->currentPtr->timestamp);
    POSE_Objects[parent->thisIndex].Step(pm);
  }
}

// Roll back to RBevent
void opt::Rollback()
{
  Event *ev = eq->currentPtr->prev, *recoveryPoint;

  RBevent = eq->RecomputeRollbackTime();
  if (!RBevent) return;
  // find earliest event that must be undone
  recoveryPoint = RBevent;
  // skip forward over other stragglers
  while ((recoveryPoint != eq->backPtr) && (recoveryPoint->done == 0)) 
    recoveryPoint = recoveryPoint->next;
  if (recoveryPoint == eq->backPtr) {
    CkPrintf("ERROR: opt::Rollback: no executed events between RBevent & backPtr.\n");
    CkExit();
  }

  // roll back over recovery point
#ifdef POSE_STATS_ON
  localStats->Rollback();
#endif
  while (ev != recoveryPoint) {     // rollback, undoing along the way
    UndoEvent(ev);                  // undo the event
    ev = ev->prev;     
  }

  // ev is now at recovery point
  if (!recoveryPoint->cpData) {     // no checkpoint, must recover state
    CkPrintf("WARNING: no cpData at recovery point.\n");
    UndoEvent(recoveryPoint);       // undo the recovery point
    RecoverState(recoveryPoint);    // recover the state prior to target
  }
  else {                            // checkpoint available, simply undo
    targetEvent = recoveryPoint;
    UndoEvent(recoveryPoint);       // undo the recovery point
  }

  eq->SetCurrentPtr(RBevent);       // adjust currentPtr
  RBevent = targetEvent = NULL;     // reset RBevent & targetEvent
}

// Undo a single event, cancelling its spawned events
void opt::UndoEvent(Event *e)
{
  if (e->done == 1) {
    currentEvent = e;
    CancelSpawn(e);                                // cancel spawned events
#ifdef POSE_STATS_ON
    localStats->Undo();
#endif
    parent->UNDOs++;
    parent->ResolveFn(((e->fnIdx) * -1), e->msg);  // execute the anti-method
    if (e->commitBfrLen > 0)                       // clean up buffered output
      free(e->commitBfr);
    e->commitBfr = NULL;
    e->done = e->commitBfrLen = 0;
    delete e->cpData;
    e->cpData = NULL;
  }
}

// Cancel events in cancellations list
void opt::CancelEvents() 
{
  Event *ev, *tmp, *recoveryPoint;
  int found, eGVT = localPVT->getGVT();
  CancelNode *it=NULL, *last=NULL;

  last = parent->cancels.GetItem(eGVT);  // make note of last item to examine
  if (!last)  // none of the cancellations are early enough to bother with
    return;
  while (!parent->cancels.IsEmpty()) {   // loop through all cancellations
    it = parent->cancels.GetItem(eGVT);
    if (!it)  // none of the cancellations are early enough to bother with
      return;
    found = 0;                         // init the found flag to not found (0)

    // search cancellations list for a cancellation that has a corresponding
    // event in the event queue
    while (!found) {  // loop until one is found, or exit fn if all examined
      ev = eq->currentPtr;               // set search start point

      // look for "it" above currentPtr
      if ((ev->timestamp <= it->timestamp) && (ev != eq->backPtr)) {
	// search forward from currentPtr until backPtr is reached or break
	while ((ev->timestamp >= 0) && (ev->timestamp <= it->timestamp)) {
	  if (ev->evID == it->evID) {    // found it
	    found = 1;
	    break;
	  }
	  else ev = ev->next;
	}
	if (!found) {             // not in linked list; check the heap
	  found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	  if (found) ev = NULL;   // make ev NULL so we know it was deleted
	}
      }
      else {                      // current is backPtr; check the heap
	found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	if (found) ev = NULL;     // make ev NULL so we know it was deleted
      }
      if (!found) {               // "it" may be below current
	// search backward from currentPtr until backPtr is reached or break
	ev = eq->currentPtr->prev;
	while ((ev->timestamp >= 0) && (ev->timestamp >= it->timestamp))
	  if (ev->evID == it->evID) {  // found it
	    found = 1;
	    break;
	  }
	  else ev = ev->prev;
      }
      if (!found) {                       // "it" event has not arrived yet
	if (it == last) {                 // seen all cancellations during this call
	  /*
	  CkPrintf("WARNING: opt::CancelEvents: Waiting for [%d.%d.%d] to arrive\n",
		   it->evID.id, it->evID.pe);
	  */
	  return;                         // print a message and bail out
	}
	it = parent->cancels.GetItem(eGVT);   // try the next cancellation
	if (!it)  // none of the cancellations are early enough to bother with
	  return;
      }
    }

    if (ev && (ev->done == 0)) {   // found it to be unexecuted; get rid of it
      if (ev == eq->currentPtr)    // adjust currentPtr
	eq->ShiftEvent();
      eq->DeleteEvent(ev);         // delete the event
    }
    else if (ev) { // it's been executed, so rollback
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(RB_TIMER);
#endif
      recoveryPoint = ev;            // ev is the target rollback point
      tmp = eq->currentPtr->prev;    
#ifdef POSE_STATS_ON
      localStats->Rollback();
#endif

      if (ev->done == 0) CkPrintf("Why am I here?\n");
      while (tmp != recoveryPoint) { // rollback, undoing along the way
	UndoEvent(tmp);              // undo the event
	tmp = tmp->prev;
      }

      if (!recoveryPoint->cpData) {       // no checkpoint, must recover state
	UndoEvent(recoveryPoint);         // undo the recovery point
	RecoverState(recoveryPoint);      // recover the state prior to target
      }
      else {                              // checkpoint available, simply undo
	targetEvent = recoveryPoint;
	UndoEvent(recoveryPoint);         // undo the recovery point
      }

      eq->SetCurrentPtr(recoveryPoint->next); // adjust currentPtr
      eq->DeleteEvent(recoveryPoint);         // delete the targetEvent

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
      parent->cancels.RemoveItem(it);                  // Clean up
      if (!parent->cancels.IsEmpty())
	last = parent->cancels.GetItem(eGVT);
    }
    else
      parent->cancels.RemoveItem(it);                  // Clean up
  } // end outer while which loops through entire cancellations list
  RBevent = eq->RecomputeRollbackTime();
}

int opt::SafeTime()
 {  // compute safe time for object
    int ovt=userObj->OVT(), theTime=-1, ec=parent->cancels.earliest,
      gvt=localPVT->getGVT(), worktime = eq->currentPtr->timestamp;
    
    if (!RBevent && (ec<0) && (worktime < 0) && (ovt <= gvt))  // idle object
      return -1;
    
    if (worktime > theTime)                         // check queued events
      theTime = worktime;
    if ((theTime >= 0) && (ovt > theTime))
      theTime = ovt;
    if (RBevent && ((RBevent->timestamp<theTime)||(theTime<0))) // rollbacks
      theTime = RBevent->timestamp;
    if ((ec >= 0) && ((ec < theTime)||(theTime<0)))     // check cancellations
      theTime = ec;
    if ((theTime < 0) && (ovt > gvt))
      theTime = ovt;
    
    if ((theTime < gvt) && (theTime >= 0)) {
      CkPrintf("WARNING: opt::SafeTime: time calculated (%d) is less than GVT estimate (%d)\n ovt=%d ec=%d worktime=%d\n", theTime, gvt, ovt, ec, worktime);
      theTime = gvt;
    }
    //CkPrintf("%d on PE %d: ovt=%d ec=%d worktime=%d\n", 
    //parent->thisIndex, CkMyPe(), ovt, ec, worktime);
    return theTime;
  }



void opt::RecoverState(Event *recoveryPoint)
{
  // PRE: rolled back over recoveryPoint. recoveryPoint should
  //      not have been checkpointed.
  Event *ev;
  if (recoveryPoint->cpData) {
    CkPrintf("WARNING: opt::RecoverState: recoveryPoint has checkpoint already.\n");
    return;
  }

  CpvAccess(stateRecovery) = 1;  // change forward execution behavior: recover state only
  // search for checkpoint
  ev = recoveryPoint->prev;
  while ((ev != eq->frontPtr) && (!ev->cpData)) {
    if (ev->commitBfrLen > 0)
      free(ev->commitBfr);
    ev->commitBfr = NULL;
    ev->commitBfrLen = 0;
    ev = ev->prev;
  }
  if (ev == eq->frontPtr) {
    CkPrintf("[%d] ERROR: opt::RecoverState: %d no prior checkpoints -- cannot recover state.\n", CkMyPe(), parent->thisIndex);
    return;
  }

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
  // userObj->CheckpointAll();
  while (ev != recoveryPoint) {
    if (ev->done == 1) {
      currentEvent = ev;
      parent->ResolveFn(ev->fnIdx, ev->msg);
    }
    ev = ev->next;
  }
  userObj->ResetCheckpointRate();
  CpvAccess(stateRecovery) = 0;  // return forward execution behavior to normal
}
