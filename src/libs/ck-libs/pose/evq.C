/// Queue of executed and unexecuted events on a poser
#include "pose.h"

/// Basic Constructor
eventQueue::eventQueue()
{
#ifdef POSE_DOP_ON
  sprintf(filename, "dop%d.log", CkMyPe());
  fp = fopen(filename, "a");
  lastLoggedVT = 0;
#endif
  Event *e;
  eqh = new EqHeap();  // create the heap for incoming events
  largest = POSE_UnsetTS;
  // create the front sentinel node
  e = new Event();
  e->timestamp = POSE_UnsetTS;
  e->done = -1;
  e->fnIdx = -99;
  e->msg = NULL;
  e->commitBfr = NULL;
  e->spawnedList = NULL;
  e->commitBfrLen = 0;
  e->next = e->prev = NULL;
  frontPtr = e;
  // create the back sentinel node
  e = new Event();
  e->timestamp=POSE_UnsetTS;
  e->done = -1;
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
  //sanitize();
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
  //sanitize();
  Event *tmp = backPtr->prev; // start at back of queue

  if (e->timestamp > largest) largest = e->timestamp;
  // check if new event should go on heap: 
  // greater than last timestamp in queue, 
  // or currentPtr is at back (presumably because heap is empty)
  //CkPrintf("Received event "); e->evID.dump(); CkPrintf(" at %d...\n", e->timestamp);
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
    if ((currentPtr->prev == e) && (currentPtr->done < 1))
      currentPtr = currentPtr->prev;
  }
  //sanitize();
}

/// Move currentPtr to next event in queue
void eventQueue::ShiftEvent() { 
  Event *e;
  //sanitize();
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
  if (currentPtr == backPtr) largest = POSE_UnsetTS;
  FindLargest();
  //sanitize();
}

/// Commit (delete) events before target timestamp ts
void eventQueue::CommitEvents(sim *obj, POSE_TimeType ts)
{
  //sanitize();
#ifdef POSE_DOP_ON
  fpos_t fptr;
  localStat *localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  Event *target = frontPtr->next, *commitPtr = frontPtr->next;
  if (ts == POSE_UnsetTS) ts = currentPtr->timestamp;  
  if (ts == POSE_UnsetTS) ts = currentPtr->prev->timestamp;  
  ts++;

  if (obj->objID->usesAntimethods()) {
    while ((commitPtr->timestamp < ts) && (commitPtr != backPtr) 
	   && (commitPtr != currentPtr)) {
      obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg); // call commit fn
#ifdef POSE_DOP_ON
      if (lastLoggedVT >= commitPtr->svt)
	commitPtr->svt = commitPtr->evt = -1;
      else lastLoggedVT = commitPtr->evt;
#if USE_LONG_TIMESTAMPS
      while (!fprintf(fp, "%f %f %lld %lld\n", commitPtr->srt, commitPtr->ert,
#else
      while (!fprintf(fp, "%f %f %d %d\n", commitPtr->srt, commitPtr->ert,
#endif
		      commitPtr->svt, commitPtr->evt)) {
	fsetpos(fp, &fptr);
      }
      fgetpos(fp, &fptr);
      localStats->SetMaximums(commitPtr->evt, commitPtr->ert);
#endif
      if (commitPtr->commitBfrLen > 0)  { // print buffered output
	CkPrintf("%s", commitPtr->commitBfr, ts, commitPtr->timestamp);
	if (commitPtr->commitErr) CmiAbort("Commit ERROR");
      }
      if (commitPtr->cpData) delete commitPtr->cpData;
      commitPtr = commitPtr->next;
      delete commitPtr->prev; // delete committed event
    }
    commitPtr->prev = frontPtr; // reattach front sentinel node
    frontPtr->next = commitPtr;
  }
  else {
    while ((target != backPtr) && (target->timestamp < ts) && 
	   (target != currentPtr)) { // commit upto ts
      while (commitPtr != target) { // commit upto next checkpoint
	CmiAssert(commitPtr->done == 1); // only commit executed events
	obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg); // call commit fn
#ifdef POSE_DOP_ON
	if (lastLoggedVT >= commitPtr->svt)
	  commitPtr->svt = commitPtr->evt = -1;
	else lastLoggedVT = commitPtr->evt;
#if USE_LONG_TIMESTAMPS
	while (!fprintf(fp, "%f %f %lld %lld\n", commitPtr->srt, commitPtr->ert,
#else	
	while (!fprintf(fp, "%f %f %d %d\n", commitPtr->srt, commitPtr->ert,
#endif
			commitPtr->svt, commitPtr->evt)) {

	  fsetpos(fp, &fptr);
	}
	fgetpos(fp, &fptr);
	localStats->SetMaximums(commitPtr->evt, commitPtr->ert);
#endif
	if (commitPtr->commitBfrLen > 0)  { // print buffered output
	  CkPrintf("%s", commitPtr->commitBfr);
	  if (commitPtr->commitErr) CmiAbort("Commit ERROR");
	}
	if (commitPtr->cpData) delete commitPtr->cpData;
	commitPtr = commitPtr->next;
	delete commitPtr->prev; // delete committed event
      }
      //find next target
      target = target->next;
      while (!target->cpData && (target->timestamp < ts) && 
	     (target != backPtr))
	target = target->next;
    }
    commitPtr->prev = frontPtr; // reattach front sentinel node
    frontPtr->next = commitPtr;
  }
  //  sanitize();
}

/// Change currentPtr to point to event e
void eventQueue::SetCurrentPtr(Event *e) { 
  Event *tmp = e;
  // Check possible error conditions
  CmiAssert((e->done == 0) || (e->done == -1)); // e is not done or a sentinel
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
  //  sanitize();
  Event *ev = frontPtr->next; // start at front
  //  while ((ev->done == 1) && (ev != currentPtr)) ev = ev->next;
  while (ev->done == 1) ev = ev->next;
  if (ev == currentPtr) return NULL; // no unexecuted events up to currentPtr
  return ev;
}

/// Delete event and reconnect surrounding events in queue
void eventQueue::DeleteEvent(Event *ev) 
{
  //sanitize();
  CmiAssert(ev != currentPtr);
  CmiAssert(ev->spawnedList == NULL);
  CmiAssert(ev != frontPtr);
  CmiAssert(ev != backPtr);
  // first connect surrounding events
  ev->prev->next = ev->next;
  ev->next->prev = ev->prev;
  POSE_TimeType ts = ev->timestamp;
  delete ev; // then delete the event
  if (ts == largest) FindLargest();
  //sanitize();
}

/// Find largest timestamp of the unexecuted events
void eventQueue::FindLargest()
{
  POSE_TimeType hs = eqh->FindMax();
  largest = backPtr->prev->timestamp;
  if (largest < hs) largest = hs;
}

/// Add id, e and ts as an entry in currentPtr's spawned list
void eventQueue::AddSpawnToCurrent(int id, eventID e, POSE_TimeType ts) 
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
#if USE_LONG_TIMESTAMPS
    CkPrintf("%lld[", e->timestamp); e->evID.dump(); CkPrintf("]");
#else
    CkPrintf("%d[", e->timestamp); e->evID.dump(); CkPrintf("]");
#endif
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

/// Check validity of data fields
void eventQueue::sanitize()
{
  // check sentinel nodes
  CmiAssert(frontPtr != NULL);
  CmiAssert(frontPtr->timestamp == POSE_UnsetTS);
  CmiAssert(frontPtr->done == -1);
  CmiAssert(frontPtr->fnIdx == -99);
  CmiAssert(frontPtr->msg == NULL);
  CmiAssert(frontPtr->commitBfr == NULL);
  CmiAssert(frontPtr->spawnedList == NULL);
  CmiAssert(frontPtr->next != NULL);
  CmiAssert(frontPtr->prev == NULL);
  CmiAssert(frontPtr->commitBfrLen == 0);
  CmiAssert(backPtr != NULL);
  CmiAssert(backPtr->timestamp == POSE_UnsetTS);
  CmiAssert(backPtr->done == -1);
  CmiAssert(backPtr->fnIdx == -100);
  CmiAssert(backPtr->msg == NULL);
  CmiAssert(backPtr->commitBfr == NULL);
  CmiAssert(backPtr->spawnedList == NULL);
  CmiAssert(backPtr->next == NULL);
  CmiAssert(backPtr->prev != NULL);
  CmiAssert(backPtr->commitBfrLen == 0);

  // traverse forward
  Event *tmp = frontPtr->next;
  while (tmp != backPtr) {
    CmiAssert(tmp->next != NULL);
    tmp->sanitize();
    tmp = tmp->next;
  }

  // traverse backward
  tmp = backPtr->prev;
  while (tmp != frontPtr) {
    CmiAssert(tmp->prev != NULL);
    tmp->sanitize();
    tmp = tmp->prev;
  }

  // check currentPtr
  CmiAssert(currentPtr != NULL);
  // should also make sure that the event this points to is in the queue!
  // traverse forward
  tmp = currentPtr;
  while (tmp != backPtr) {
    CmiAssert(tmp->next != NULL);
    tmp = tmp->next;
  } // tmp is now at backptr
  // traverse backward to currentPtr
  while (tmp != currentPtr) {
    CmiAssert(tmp->prev != NULL);
    tmp = tmp->prev;
  } // tmp is now at currentPtr
  // traverse backward to frontPtr
  while (tmp != frontPtr) {
    CmiAssert(tmp->prev != NULL);
    tmp = tmp->prev;
  } // tmp is now at frontPtr
  // traverse forward to currentPtr
  while (tmp != currentPtr) {
    CmiAssert(tmp->next != NULL);
    tmp = tmp->next;
  } // tmp is now at currentPtr again

  // first event in queue should always have a checkpoint
  if ((frontPtr->next != backPtr) && (frontPtr->next->done))
    CmiAssert(frontPtr->next->cpData);

  // check eqheap
  eqh->sanitize();
}
