// File: evq.C
// Defines: eventQueue class and methods
// Last Modified: 06.04.01 by Terry L. Wilmarth

#include "pose.h"

// eventQueue constructor: creates front and back nodes, connects them, 
// and inits pointers.
eventQueue::eventQueue()
{
  Event *e;

  //eqCount = 0;
  eqh = new EqHeap();  // create the heap for incoming events

  // create the front sentinel node; initialize data fields
  e = new Event();
  e->timestamp = e->done = e->evID.pe = -1;
  e->fnIdx = -99;
  e->msg = NULL;
  e->commitBfr = NULL;
  e->spawnedList = NULL;
  e->commitBfrLen = 0;
  e->next = e->prev = NULL;
  frontPtr = e;
  // create the back sentinel node; initialize data fields
  e = new Event();
  e->timestamp = e->done = e->evID.pe = -1;
  e->fnIdx = -100;
  e->msg = NULL;
  e->commitBfr = NULL;
  e->spawnedList = NULL;
  e->commitBfrLen = 0;
  e->next = e->prev = NULL;
  currentPtr = backPtr = e;  // when no unprocessed events, currentPtr=backPtr
  // link them together
  frontPtr->next = backPtr;
  backPtr->prev = frontPtr;
}

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

// Insert e in timestamp order.  If executed events with same timestamp exist,
// insert e at the back of these.
void eventQueue::InsertEvent(Event *e)
{
  Event *tmp = backPtr->prev;               // start at back of queue

  //eqCount++;
  //CkPrintf("<%d> ", eqCount);
  // check if new event should go on heap: greater than last timestamp in queue, 
  // or currentPtr is at back (presumably because heap is empty)
  if ((tmp->timestamp <= e->timestamp) && (currentPtr != backPtr))
    eqh->InsertEvent(e);                    // insert in heap
  else { // tmp->timestamp > e->timestamp; insert in linked list
    if ((currentPtr != backPtr) && (currentPtr->timestamp > e->timestamp))
      tmp = currentPtr;                     // may be closer to insertion point
    while (tmp->timestamp > e->timestamp)   // search for position
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

// Move currentPtr foward to next event
void eventQueue::ShiftEvent() { 
  Event *e;
  if (currentPtr->next)
    currentPtr = currentPtr->next;             // set currentPtr to next event
  else                                         // lost a sentinel node!
    CkPrintf("ERROR: shifting off end of event queue.\n");
  if ((currentPtr == backPtr) && (eqh->top)) { // currentPtr on back sentinel
    e = eqh->GetAndRemoveTopEvent();           // get next event from heap
    // insert event in list
    e->prev = currentPtr->prev;
    e->next = currentPtr;
    currentPtr->prev = e;
    e->prev->next = e;
    currentPtr = e;
  }
}

// Commit events before target
void eventQueue::CommitEvents(sim *obj, int ts)
{
  Event *target;
  if (ts >= 0) { // commit up to ts
    target = currentPtr->prev;
    while (target->timestamp >= ts)
      target = target->prev;
    target = target->next;
    while (!target->cpData && (target != frontPtr))
      target = target->prev;
    if (target == frontPtr) return;  // nothing to commit
    commitPtr = frontPtr->next;
  }
  else if (ts == -1) {
    commitPtr = frontPtr->next;
    if (commitPtr == currentPtr) return;  // nothing to commit
    target = currentPtr;
  }
  while (commitPtr != target) { // commit up to next checkpoint
    CmiAssert(commitPtr->done == 1);  // only commit executed events
    obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg); // commit fn
    if (commitPtr->commitBfrLen > 0)  { // print buffered I/O
      CkPrintf("%s", commitPtr->commitBfr);
      //free(commitPtr->commitBfr);
      if (commitPtr->commitErr) CmiAbort("Commit ERROR");
    }
    commitPtr = commitPtr->next;
    /* if (obj->recycCount < 0)
       obj->recyc[obj->recycCount] = commitPtr->prev->cpData;
       else*/ 
    if (commitPtr->prev->cpData)
      delete commitPtr->prev->cpData; 
    delete commitPtr->prev;  // delete committed event
  }
  commitPtr->prev = frontPtr;  // reattach front sentinel node
  frontPtr->next = commitPtr;
  commitPtr = NULL;
}

// Sets currentPtr to another event; DANGER: use carefully
void eventQueue::SetCurrentPtr(Event *e) { 
  Event *tmp = e;

  // Check various possible error conditions and whine loudly if they occur
  if (e->done == 1)
    CkPrintf("ERROR: pointing currentPtr at executed event\n");
  else if (currentPtr->done == 2)
    CkPrintf("ERROR: moving currentPtr away from executing event\n");
  else 
    while (tmp != backPtr) {
      tmp = tmp->next;
      if (tmp->done == 1)
	CkPrintf("ERROR: placing currentPtr before executed events\n");
    }
  currentPtr = e;                               // move currentPtr to e
  if ((currentPtr == backPtr) && (eqh->top)) {  // moved currentPtr to backPtr
    // get next event from heap and put it in the queue
    tmp = eqh->GetAndRemoveTopEvent();
    tmp->next = currentPtr;
    tmp->prev = currentPtr->prev;
    currentPtr->prev->next = tmp;
    currentPtr->prev = tmp;
    currentPtr = tmp;
  }
}

// Finds and returns first (earliest) unexecuted event before currentPtr
Event *eventQueue::RecomputeRollbackTime() 
{
  Event *ev = frontPtr->next;          // start at front
  
  while ((ev->done == 1) && (ev != currentPtr))
    ev = ev->next;
  if (ev == currentPtr)                // no unexecuted events up to currentPtr
    return NULL;
  return ev;
}

// Delete ev and make sure to reconnect list; called from cancel & rollback
void eventQueue::DeleteEvent(Event *ev) 
{
  if (ev == currentPtr)     // really really don't want to delete currentPtr
    CkPrintf("ERROR: deleting currentPtr.\n");
  if (ev->spawnedList)      // if event was executed, it should be undone first
    CkPrintf("ERROR: deleting event with non-empty spawnedList.\n");
  // first connect surrounding events
  ev->prev->next = ev->next;
  ev->next->prev = ev->prev;
  delete ev;                // then delete the event
  //  eqCount--; 
  //  CkPrintf("<%d> ", eqCount);
}

// Add id, e and ts as an entry in currentPtr's spawned list
void eventQueue::AddSpawnToCurrent(int id, eventID e, int ts) 
{
  SpawnedEvent *newnode = new SpawnedEvent(id, e, ts, currentPtr->spawnedList);
  if (currentPtr->done != 2) {
    CkPrintf("ERROR: eventQueue::AddSpawnToCurrent: adding spawn to non-executing event!!\n"); 
  }
  currentPtr->spawnedList = newnode;
}

// Return the first entry in currentPtr's spawned list and remove it
SpawnedEvent *eventQueue::GetNextCurrentSpawn() 
{
  SpawnedEvent *tmp = currentPtr->spawnedList;
  if (tmp) currentPtr->spawnedList = tmp->next;
  return tmp;
}

// Print contents of eventQueue to stdout (charm++ output)
void eventQueue::dump(int pdb_level)
{
  Event *e = frontPtr;

  pdb_indent(pdb_level);
  CkPrintf("[EVENTQUEUE: \n");
  pdb_indent(pdb_level);
  while (e) {
    CkPrintf("%d[", e->timestamp); e->evID.dump(); CkPrintf("]");
    if (e == frontPtr) CkPrintf("(FP)");
    if (e == currentPtr) CkPrintf("(CP)");
    if (e == backPtr) CkPrintf("(BP)");
    CkPrintf(" ");
    e = e->next;
  }
  CkPrintf("\n");
  eqh->dump(pdb_level+1);
  pdb_indent(pdb_level);
  CkPrintf("end EVENTQUEUE]\n");
}

// Pup the entire event queue
void eventQueue::pup(PUP::er &p) 
{
  // PRE: this has been initialized, so front & back Ptrs exist, and eqh has 
  // been initialized
  Event *tmp;
  int countlist = 0;
  
  if (p.isUnpacking()) { // UNPACKING
    p(countlist);                      // unpack count of events in list
    tmp = frontPtr;                    // front & backPtrs should already exist
    for (int i=0; i<countlist; i++) {  // unpack countlist events
      tmp->next = new Event;
      tmp->next->prev = tmp;
      tmp->next->next = NULL;
      tmp = tmp->next;
      tmp->pup(p);
    }
    tmp->next = backPtr;               // reattach backptr
    backPtr->prev = tmp;
    currentPtr = backPtr;              // reposition currentPtr
    if ((countlist > 0) && (backPtr->prev != frontPtr))  
      while ((currentPtr->prev->done == 0) && 
	     (currentPtr->prev != frontPtr))
	currentPtr = currentPtr->prev;
    eqh->pup(p);           // unpack the heap (pre-allocated in call to eq pup)
  }
  else { // PACKING & SIZING
    tmp = frontPtr->next;
    while (tmp != backPtr) {           // count events in list
      countlist++;
      tmp = tmp->next;
    }
    p(countlist);                      // pack event count
    tmp = frontPtr->next;
    for (int i=0; i<countlist; i++) {  // pack each event
      tmp->pup(p);
      tmp = tmp->next;
    }
    eqh->pup(p);                       // pack the heap
  }
}
