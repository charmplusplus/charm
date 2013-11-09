// File: con.C
// Module for conservative simulation strategy class
// Last Modified: 07.31.01 by Terry L. Wilmarth

#include "pose.h"

void con::Step()
{
  Event *ev;

#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(CAN_TIMER);
#endif
  if (!parent->cancels.IsEmpty())  // Cancel as much as possible
    CancelEvents();
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);
#endif
  // Prepare to execute an event
  ev = eq->currentPtr;
  while ((ev->timestamp <= localPVT->getGVT()) && (ev->timestamp >= 0)) { 
    // execute all events at GVT
    currentEvent = ev;
    ev->done = 2;
    parent->DOs++;
#if !CMK_TRACE_DISABLED
    if(pose_config.stats){
      localStats->Do();
      localStats->SwitchTimer(DO_TIMER);
    }
#endif
      parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
#if !CMK_TRACE_DISABLED
      if(pose_config.stats)
	localStats->SwitchTimer(SIM_TIMER);
#endif
      ev->done = 1;
      eq->ShiftEvent();                       // move on to next event
      ev = eq->currentPtr;
  }
}

// Cancel events in cancellations list
void con::CancelEvents() 
{
  Event *ev;
  int found;
  POSE_TimeType eGVT = localPVT->getGVT();
  CancelNode *it, *last;

  last = parent->cancels.GetItem();     // make note of last item to be examined
  if (!last)  // none of the cancellations are early enough to bother with
    return;
  while (!parent->cancels.IsEmpty()) {      // loop through all cancellations
    it = parent->cancels.GetItem();     // it is the event to cancel
    if (!it)  // none of the cancellations are early enough to bother with
      return;
    found = 0;                              // init the found flag to not found (0)
    
    // search the cancellations list for a cancellation that has a corresponding
    // event in the event queue
    while (!found) {  // loop until one is found, or exit fn if all examined
      ev = eq->currentPtr;                  // set search start point

      // look for "it" above currentPtr
      if ((ev->timestamp <= it->timestamp) && (ev != eq->back())) {
	while ((ev->timestamp >= 0) && (ev->timestamp <= it->timestamp))
	  if (ev->evID == it->evID) {
	    found = 1;
	    break;
	  }
	  else ev = ev->next;
	if (!found) {                  // not in linked list; check heap
	  found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	  if (found) ev = NULL;        // make ev NULL so we know it was deleted
	}
      }
      else if (ev != eq->back()) {    // current at back of queue; check the heap
	found = eq->eqh->DeleteEvent(it->evID, it->timestamp);
	if (found) ev = NULL;          // make ev NULL so we know it was deleted
      }
      else if (ev->timestamp > it->timestamp) {  // ERROR: the event is a past event
	CkPrintf("ERROR: con::CancelEvents: Trying to cancel past event.\n");
	CkExit();
      }
      if (!found) {                    // "it" event has not arrived yet
	if (it == last) {              // seen all cancellations during this call
	  /*
	  CkPrintf("WARNING: con::CancelEvents: Waiting for [%d.%d] to arrive\n",
		   it->evID.id, it->evID.pe);
	  */
	  return;
	}
	it = parent->cancels.GetItem();
	if (!it)  // none of the cancellations are early enough to bother with
	  return;
      }
    }
      
    if (ev && (ev->done == 0)) {            // found it; get rid of it
      if (ev == eq->currentPtr)             // adjust currentPtr
	eq->ShiftEvent();
      eq->DeleteEvent(ev);                  // delete the event
    }
    else if (ev) {                          // ERROR: event was executed
      CkPrintf("ERROR: con::CancelEvents: Trying to cancel executed event.\n");
      CkExit();
    }

    parent->cancels.RemoveItem(it);         // clean up
  }
}


