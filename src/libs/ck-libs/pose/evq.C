/// Queue of executed and unexecuted events on a poser
#include "pose.h"

/// Basic Constructor
eventQueue::eventQueue()
{
  Event *e;
  eqh = new EqHeap();  // create the heap for incoming events
  // create the front sentinel node
  e = new Event();
  e->timestamp = e->done = -1;
  e->fnIdx = -99;
  e->msg = NULL;
  e->commitBfr = NULL;
  e->spawnedList = NULL;
  e->commitBfrLen = 0;
  e->next = e->prev = NULL;
  frontPtr = e;
  // create the back sentinel node
  e = new Event();
  e->timestamp = e->done = -1;
  e->fnIdx = -100;
  e->msg = NULL;
  e->commitBfr = NULL;
  e->spawnedList = NULL;
  e->commitBfrLen = 0;
  e->next = e->prev = NULL;
  currentPtr = backPtr = e; // when no unprocessed events, currentPtr=backPtr
  // link them together
  frontPtr->next = backPtr;
  backPtr->prev = frontPtr;
}

/// Destructor
eventQueue::~eventQueue()
{
  Event *tmp1 = frontPtr, *tmp2 = frontPtr->next;
  while (tmp2) {
    free(tmp1);
    tmp1 = tmp2;
    tmp2 = tmp1->next;
  }
  free(tmp1);
  delete eqh;
}

/// Insert e in the queue in timestamp order
void eventQueue::InsertEvent(Event *e)
{
  Event *tmp = backPtr->prev; // start at back of queue

  // check if new event should go on heap: 
  // greater than last timestamp in queue, 
  // or currentPtr is at back (presumably because heap is empty)
  if ((tmp->timestamp <= e->timestamp) && (currentPtr != backPtr))
    eqh->InsertEvent(e); // insert in heap
  else { // tmp->timestamp > e->timestamp; insert in linked list
    if ((currentPtr != backPtr) && (currentPtr->timestamp > e->timestamp))
      tmp = currentPtr; // may be closer to insertion point
    while (tmp->timestamp > e->timestamp) // search for position
      tmp = tmp->prev;
    // insert e
    e->prev = tmp;
    e->next = tmp->next;
    e->next->prev = e;
    tmp->next = e;
    // if e is inserted before currPtr, move currPtr back to avoid rollback
    if ((currentPtr->prev == e) && (currentPtr->done <= 0))
      currentPtr = currentPtr->prev;
  }
}

/// Move currentPtr to next event in queue
void eventQueue::ShiftEvent() { 
  Event *e;
  CmiAssert(currentPtr->next != NULL);
  currentPtr = currentPtr->next; // set currentPtr to next event
  if ((currentPtr == backPtr) && (eqh->top)) { // currentPtr on back sentinel
    e = eqh->GetAndRemoveTopEvent(); // get next event from heap
    // insert event in list
    e->prev = currentPtr->prev;
    e->next = currentPtr;
    currentPtr->prev = e;
    e->prev->next = e;
    currentPtr = e;
  }
}

/// Commit (delete) events before target timestamp ts
void eventQueue::CommitEvents(sim *obj, int ts)
{
  Event *target, *commitPtr;
  if (ts >= 0) { // commit up to ts
    target = currentPtr->prev;
    while (target->timestamp >= ts)
      target = target->prev;
    target = target->next;
    while (!target->cpData && (target != frontPtr))
      target = target->prev;
    if (target == frontPtr) return; // nothing to commit
    commitPtr = frontPtr->next;
  }
  else if (ts == -1) {
    commitPtr = frontPtr->next;
    if (commitPtr == currentPtr) return; // nothing to commit
    target = currentPtr;
  }
  while (commitPtr != target) { // commit up to next checkpoint
    CmiAssert(commitPtr->done == 1); // only commit executed events
    obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg); // call commit fn
    if (commitPtr->commitBfrLen > 0)  { // print buffered output
      CkPrintf("%s", commitPtr->commitBfr);
      if (commitPtr->commitErr) CmiAbort("Commit ERROR");
    }
    commitPtr = commitPtr->next;
    if (commitPtr->prev->cpData) delete commitPtr->prev->cpData; 
    delete commitPtr->prev; // delete committed event
  }
  commitPtr->prev = frontPtr; // reattach front sentinel node
  frontPtr->next = commitPtr;
  commitPtr = NULL;
}

/// Change currentPtr to point to event e
void eventQueue::SetCurrentPtr(Event *e) { 
  Event *tmp = e;
  // Check possible error conditions
  CmiAssert(e->done == 0);
  CmiAssert(currentPtr->done != 2);
  currentPtr = e; // move currentPtr to e
  if ((currentPtr == backPtr) && (eqh->top)) { // moved currentPtr to backPtr
    // get next event from heap and put it in the queue
    tmp = eqh->GetAndRemoveTopEvent();
    tmp->next = currentPtr;
    tmp->prev = currentPtr->prev;
    currentPtr->prev->next = tmp;
    currentPtr->prev = tmp;
    currentPtr = tmp;
  }
}

/// Return first (earliest) unexecuted event before currentPtr
Event *eventQueue::RecomputeRollbackTime() 
{
  Event *ev = frontPtr->next; // start at front
  while ((ev->done == 1) && (ev != currentPtr)) ev = ev->next;
  if (ev == currentPtr) return NULL; // no unexecuted events up to currentPtr
  return ev;
}

/// Delete event and reconnect surrounding events in queue
void eventQueue::DeleteEvent(Event *ev) 
{
  CmiAssert(ev != currentPtr);
  CmiAssert(ev->spawnedList == NULL);
  // first connect surrounding events
  ev->prev->next = ev->next;
  ev->next->prev = ev->prev;
  delete ev; // then delete the event
}

/// Add id, e and ts as an entry in currentPtr's spawned list
void eventQueue::AddSpawnToCurrent(int id, eventID e, int ts) 
{
  SpawnedEvent *newnode = new SpawnedEvent(id, e, ts, currentPtr->spawnedList);
  CmiAssert(currentPtr->done == 2);
  currentPtr->spawnedList = newnode;
}

/// Return the first entry in currentPtr's spawned list and remove it
SpawnedEvent *eventQueue::GetNextCurrentSpawn() 
{
  SpawnedEvent *tmp = currentPtr->spawnedList;
  if (tmp) currentPtr->spawnedList = tmp->next;
  return tmp;
}

/// Dump the event queue
void eventQueue::dump()
{
  Event *e = frontPtr;
  CkPrintf("[EVENTQUEUE: \n");
  while (e) {
    CkPrintf("%d[", e->timestamp); e->evID.dump(); CkPrintf("]");
    if (e == frontPtr) CkPrintf("(FP)");
    if (e == currentPtr) CkPrintf("(CP)");
    if (e == backPtr) CkPrintf("(BP)");
    CkPrintf(" ");
    e = e->next;
  }
  CkPrintf("\n");
  eqh->dump();
  CkPrintf("end EVENTQUEUE]\n");
}

/// Pack/unpack/sizing operator
void eventQueue::pup(PUP::er &p) 
{
  Event *tmp;
  int countlist = 0;
  if (p.isUnpacking()) { // UNPACKING
    p(countlist); // unpack count of events in list
    tmp = frontPtr; // front & backPtrs should already exist
    for (int i=0; i<countlist; i++) { // unpack countlist events
      tmp->next = new Event;
      tmp->next->prev = tmp;
      tmp->next->next = NULL;
      tmp = tmp->next;
      tmp->pup(p);
    }
    tmp->next = backPtr; // reattach backptr
    backPtr->prev = tmp;
    currentPtr = backPtr; // reposition currentPtr
    if ((countlist > 0) && (backPtr->prev != frontPtr))  
      while ((currentPtr->prev->done == 0) && 
	     (currentPtr->prev != frontPtr))
	currentPtr = currentPtr->prev;
    eqh->pup(p); // unpack the heap (pre-allocated in call to eq pup)
  }
  else { // PACKING & SIZING
    tmp = frontPtr->next;
    while (tmp != backPtr) { // count events in list
      countlist++;
      tmp = tmp->next;
    }
    p(countlist); // pack event count
    tmp = frontPtr->next;
    for (int i=0; i<countlist; i++) { // pack each event
      tmp->pup(p);
      tmp = tmp->next;
    }
    eqh->pup(p); // pack the heap
  }
}
