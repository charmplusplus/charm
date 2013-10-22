/// Queue of executed and unexecuted events on a poser
#include "pose.h"

/// Basic Constructor
eventQueue::eventQueue()
{
  lastLoggedVT = 0;
  Event *e;
  eqh = new EqHeap();  // create the heap for incoming events
  largest = POSE_UnsetTS;
  mem_usage = 0;
  eventCount = 0;
  tsOfLastInserted = 0;
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
  RBevent = NULL;
  // other variables
  recentAvgEventSparsity = 1;
  sparsityStartTime = 0;
  sparsityCalcCount = 0;
  tsDiffCount = 0;
  lastCommittedTS = 0;
  for (int i = 0; i < DIFFS_TO_STORE; i++) {
    tsCommitDiffs[i] = 0;
  }
#ifdef MEM_TEMPORAL
  localTimePool = (TimePool *)CkLocalBranch(TempMemID);
#endif
#ifdef EQ_SANITIZE
  sanitize();
#endif
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
  tsOfLastInserted = e->timestamp;
  if(pose_config.deterministic)
    {
      InsertEventDeterministic(e);
    }
  else
    {
#ifdef EQ_SANITIZE
      sanitize();
#endif
      Event *tmp = backPtr->prev; // start at back of queue

      if (e->timestamp > largest) largest = e->timestamp;
      eventCount++;
      // check if new event should go on heap: 
      // greater than last timestamp in queue, 
      // or currentPtr is at back (presumably because heap is empty)
      //CkPrintf("Received event "); e->evID.dump(); CkPrintf(" at %d...\n", e->timestamp);
      if ((tmp->timestamp < e->timestamp || (tmp->timestamp == e->timestamp && tmp->evID < e->evID)) && (currentPtr != backPtr))
	eqh->InsertEvent(e); // insert in heap
      else { // tmp->timestamp > e->timestamp; insert in linked list
	if ((currentPtr != backPtr) && (currentPtr->timestamp > e->timestamp))
	  tmp = currentPtr; // may be closer to insertion point
	while (tmp->timestamp > e->timestamp || (tmp->timestamp == e->timestamp && tmp->evID > e->evID)) // search for position
	  tmp = tmp->prev;
	// insert e
	e->prev = tmp;
	e->next = tmp->next;
	e->next->prev = e;
	tmp->next = e;
	// if e is inserted before currPtr, move currPtr back to avoid rollback
	if ((currentPtr->prev == e) && (currentPtr->done < 1))
	  currentPtr = currentPtr->prev;
	else if ((currentPtr == backPtr) || (e->timestamp < currentPtr->timestamp || (e->timestamp == currentPtr->timestamp && e->evID < currentPtr->evID)))
	  SetRBevent(e);
      }
#ifdef EQ_SANITIZE
      sanitize();
#endif
    }
}

/// Insert e in the queue in timestamp order
/** If executed events with same timestamp exist, sort e into them based on
    eventID */
void eventQueue::InsertEventDeterministic(Event *e)
{
  Event *tmp = backPtr->prev; // start at back of queue
  if (e->timestamp > largest) largest = e->timestamp;
  eventCount++;
  // check if new event should go on heap: 
  // greater than last timestamp in queue, 
  // or currentPtr is at back (presumably because heap is empty)
  //CkPrintf("Received event "); e->evID.dump(); CkPrintf(" at %d...\n", e->timestamp);
  if ((tmp->timestamp < e->timestamp || (tmp->timestamp == e->timestamp && tmp->evID < e->evID)) && (currentPtr != backPtr))
    eqh->InsertEvent(e); // insert in heap
  else { // tmp->timestamp > e->timestamp; insert in linked list
    if ((currentPtr != backPtr) && (currentPtr->timestamp > e->timestamp))
      tmp = currentPtr; // may be closer to insertion point
    while (tmp->timestamp > e->timestamp || (tmp->timestamp == e->timestamp && tmp->evID > e->evID)) // search for position
      tmp = tmp->prev;
    // tmp now points to last event with timestamp <= e's
    if (tmp->timestamp == e->timestamp) // deterministic bit
      while ((tmp->timestamp == e->timestamp) && (e->evID < tmp->evID))
	tmp = tmp->prev;
    // insert e
    e->prev = tmp;
    e->next = tmp->next;
    e->next->prev = e;
    tmp->next = e;
    // if e is inserted before currPtr, move currPtr back to avoid rollback
    if ((currentPtr->prev == e) && (currentPtr->done < 1))
      currentPtr = currentPtr->prev;
    else if ((currentPtr == backPtr) || 
	     ((e->timestamp < currentPtr->timestamp) ||
	      ((e->timestamp == currentPtr->timestamp) && 
	       (e->evID < currentPtr->evID))))
      SetRBevent(e);
  }
#ifdef EQ_SANITIZE
  sanitize();
#endif
}

void eventQueue::CommitStatsHelper(sim *obj, Event *commitPtr) {
#if !CMK_TRACE_DISABLED
  localStat *localStats = (localStat *)CkLocalBranch(theLocalStats);
  if (pose_config.stats) {
    localStats->Commit();
  }

  if (pose_config.dop) {
    // if more than one event occurs at the same virtual time on this object, 
    // only count the first event
    if (lastLoggedVT >= commitPtr->svt) {
      commitPtr->svt = commitPtr->evt = -1;
    } else {
      lastLoggedVT = commitPtr->evt;
    }
    localStats->WriteDopData(commitPtr->srt, commitPtr->ert, commitPtr->svt, commitPtr->evt);
    localStats->SetMaximums(commitPtr->evt, commitPtr->ert);
  }
#endif

  // only execute if sync strategy is adapt5
  if (obj->myStrat->STRAT_T == ADAPT5_T) {

    sparsityCalcCount++;
    // run the sparsity calculation and store in strat
    if (sparsityCalcCount >= EVQ_SPARSE_CALC_PERIOD) {
      recentAvgEventSparsity = (int)((commitPtr->timestamp - sparsityStartTime) / sparsityCalcCount);
      if (recentAvgEventSparsity < 1) {
	recentAvgEventSparsity = 1;
      }
      ((adapt5 *)obj->myStrat)->setRecentAvgEventSparsity(recentAvgEventSparsity);
      sparsityCalcCount = 0;
      sparsityStartTime = commitPtr->timestamp;
    }

    // calculate the diffs for one of the adapt5 algorithms
    POSE_TimeType diff = commitPtr->timestamp - lastCommittedTS;
    lastCommittedTS = commitPtr->timestamp;
    for (int i = 0; i < DIFFS_TO_STORE; i++) {
      if (diff > tsCommitDiffs[i]) {
	// insert in list
	for (int j = DIFFS_TO_STORE - 2; j > i; j--) {
	  tsCommitDiffs[j+1] = tsCommitDiffs[j];
	}
	if (i < DIFFS_TO_STORE - 1) {
	  // not at the end of the array
	  tsCommitDiffs[i+1] = tsCommitDiffs[i];
	}
	tsCommitDiffs[i] = diff;
	break;
      }
    }
    tsDiffCount++;

    // calculate the timeleash and store in strat every TS_DIFF_WIN_SIZE events
    if (tsDiffCount >= TS_DIFF_WIN_SIZE) {
      POSE_TimeType totalDiff = 0;
      for (int i = HIGHEST_DIFFS_TO_IGNORE; i < DIFFS_TO_STORE; i++) {
	totalDiff += tsCommitDiffs[i];
      }
      POSE_TimeType avgDiff = totalDiff / NUM_DIFFS_TO_AVERAGE;
      ((adapt5 *)obj->myStrat)->setTimeLeash(DIFFS_TO_STORE * avgDiff);
      tsDiffCount = 0;
      for (int i = 0; i < DIFFS_TO_STORE; i++) {
	tsCommitDiffs[i] = 0;
      }
    }

  }

}

/// Commit (delete) events before target timestamp ts
void eventQueue::CommitEvents(sim *obj, POSE_TimeType ts)
{
#ifdef EQ_SANITIZE
  sanitize();
#endif
  Event *target = frontPtr->next, *commitPtr = frontPtr->next;
  if (ts == POSE_endtime) {
    CommitAll(obj);
#ifdef MEM_TEMPORAL
    localTimePool->set_min_time(ts);
    localTimePool->empty_recycle_bin();
#endif
    return;
  }

  // first commit the events
  if (obj->objID->usesAntimethods()) {
    while ((commitPtr->timestamp < ts) && (commitPtr != backPtr) 
	   && (commitPtr != currentPtr)) {
      CmiAssert(commitPtr->done == 1); // only commit executed events
      obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg); // call commit fn
      obj->basicStats[0]++;
      CommitStatsHelper(obj, commitPtr);
      if (commitPtr->commitBfrLen > 0)  { // print buffered output
	CkPrintf("%s", commitPtr->commitBfr);
	if (commitPtr->commitErr) CmiAbort("Commit ERROR");
      }
      commitPtr = commitPtr->next;
    }
  }
  else {
    while ((target != backPtr) && (target->timestamp < ts) && 
	   (target != currentPtr)) { // commit upto ts
      while (commitPtr != target) { // commit upto next checkpoint
	CmiAssert(commitPtr->done == 1); // only commit executed events
	obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg);
	obj->basicStats[0]++;
	CommitStatsHelper(obj, commitPtr);
	if (commitPtr->commitBfrLen > 0)  { // print buffered output
	  CkPrintf("%s", commitPtr->commitBfr);
	  if (commitPtr->commitErr) CmiAbort("Commit ERROR");
	}
	commitPtr = commitPtr->next;
      }
      //find next target
      target = target->next;
#ifdef MEM_TEMPORAL      
      while (!target->serialCPdata && (target->timestamp <ts) && (target != backPtr))
#else
      while (!target->cpData && (target->timestamp <ts) && (target != backPtr))
#endif
	target = target->next;
    }
  }
  // now free up the memory
  Event *link = commitPtr;
  commitPtr = commitPtr->prev;
  while (commitPtr != frontPtr) {
#ifdef MEM_TEMPORAL
    if (commitPtr->serialCPdata) {
      localTimePool->tmp_free(commitPtr->timestamp, commitPtr->serialCPdata);
#else
    if (commitPtr->cpData) {
      delete commitPtr->cpData;
#endif
    }
    commitPtr = commitPtr->prev;
    delete commitPtr->next;
    mem_usage--;
  }
  frontPtr->next = link;
  link->prev = frontPtr;
#ifdef EQ_SANITIZE
  sanitize();
#endif
}

/// Commit (delete) all events
void eventQueue::CommitAll(sim *obj)
{
#ifdef EQ_SANITIZE
  sanitize();
#endif
  Event *commitPtr = frontPtr->next;
  
  // commit calls for done events
  while (commitPtr != backPtr) {
    if (commitPtr->done) {
      obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg);
      obj->basicStats[0]++;
      CommitStatsHelper(obj, commitPtr);
      if (commitPtr->commitBfrLen > 0)  { // print buffered output
	CkPrintf("%s", commitPtr->commitBfr);
	if (commitPtr->commitErr) CmiAbort("Commit ERROR");
      }
    }
    commitPtr = commitPtr->next;
  }

  // now free up the memory (frees ALL events in the queue, not just
  // the ones that are done)
  Event *link = commitPtr;
  commitPtr = commitPtr->prev;
  while (commitPtr != frontPtr) {
#ifdef MEM_TEMPORAL
    if (commitPtr->serialCPdata) {
      localTimePool->tmp_free(commitPtr->timestamp, commitPtr->serialCPdata);
    }
#else
    if (commitPtr->cpData) {
      delete commitPtr->cpData;
    }
#endif
    commitPtr = commitPtr->prev;
    mem_usage--;
    delete commitPtr->next;
  }
  frontPtr->next = link;
  link->prev = frontPtr; 
#ifdef EQ_SANITIZE
  sanitize();
#endif
}

/// Commit (delete) all events that are done (used in sequential mode)
void eventQueue::CommitDoneEvents(sim *obj) {
#ifdef EQ_SANITIZE
  sanitize();
#endif
  Event *commitPtr = frontPtr->next;
  
  // commit calls for done events
  while (commitPtr != backPtr) {
    if (commitPtr->done) {
      obj->ResolveCommitFn(commitPtr->fnIdx, commitPtr->msg);
      obj->basicStats[0]++;
      CommitStatsHelper(obj, commitPtr);
      if (commitPtr->commitBfrLen > 0)  { // print buffered output
	CkPrintf("%s", commitPtr->commitBfr);
	if (commitPtr->commitErr) CmiAbort("Commit ERROR");
      }
    }
    commitPtr = commitPtr->next;
  }

  // now free up the memory (only delete events that are done)
  Event *link = commitPtr;
  commitPtr = commitPtr->prev;
  while (commitPtr != frontPtr) {
    if (commitPtr->done == 1) {
#ifdef MEM_TEMPORAL
      if (commitPtr->serialCPdata) {
	localTimePool->tmp_free(commitPtr->timestamp, commitPtr->serialCPdata);
      }
#else
      if (commitPtr->cpData) {
	delete commitPtr->cpData;
      }
#endif
    }
    commitPtr = commitPtr->prev;
    if (commitPtr->next->done == 1) {
      mem_usage--;
      delete commitPtr->next;
    } else {
      link = commitPtr->next;
    }
  }
  frontPtr->next = link;
  link->prev = frontPtr; 
#ifdef EQ_SANITIZE
  sanitize();
#endif
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

/// Delete event and reconnect surrounding events in queue
void eventQueue::DeleteEvent(Event *ev) 
{
#ifdef EQ_SANITIZE
  sanitize();
#endif
  Event *tmp;
  CmiAssert(ev != currentPtr);
  CmiAssert(ev->spawnedList == NULL);
  CmiAssert(ev != frontPtr);
  CmiAssert(ev != backPtr);
  // if ev is earliest straggler, see if there is another
  if (RBevent == ev) {
    RBevent = NULL;
    tmp = ev->next;
    while ((tmp != currentPtr) && (tmp != backPtr) && (tmp->done == 1))
      tmp = tmp->next;
    if ((tmp != currentPtr) && (tmp != backPtr) && (tmp->done == 0))
      RBevent = tmp;
  }
  // connect surrounding events
  ev->prev->next = ev->next;
  ev->next->prev = ev->prev;
  POSE_TimeType ts = ev->timestamp;
  if (!ev->done) eventCount--;
  else mem_usage--;
  delete ev; // then delete the event
  if (ts == largest) FindLargest();
#ifdef EQ_SANITIZE
  sanitize();
#endif
}

/// Dump the event queue
void eventQueue::dump()
{
  Event *e = frontPtr;
  CkPrintf("[EVENTQUEUE: \n");
  while (e) {
#if USE_LONG_TIMESTAMPS
    CkPrintf("%lld[", e->timestamp); e->evID.dump(); CkPrintf(".%d", e->done); CkPrintf("]");
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

/// Dump the event queue to a string
char *eventQueue::dumpString() {
  Event *e = frontPtr;
  char *str= new char[PVT_DEBUG_BUFFER_LINE_LENGTH];
  char *tempStr= new char[PVT_DEBUG_BUFFER_LINE_LENGTH];
  snprintf(str, PVT_DEBUG_BUFFER_LINE_LENGTH, "[EVQ: ");
  while (e) {
#if USE_LONG_TIMESTAMPS
    snprintf(tempStr, PVT_DEBUG_BUFFER_LINE_LENGTH, "%lld[%u.%d.%d]", e->timestamp, e->evID.id, e->evID.getPE(), e->done);
#else
    sprintf(tempStr, PVT_DEBUG_BUFFER_LINE_LENGTH, "%d[%u.%d.%d]", e->timestamp, e->evID.id, e->evID.getPE(), e->done);
#endif
    strncat(str, tempStr,PVT_DEBUG_BUFFER_LINE_LENGTH);

    if (e == frontPtr) strncat(str, "(FP)", PVT_DEBUG_BUFFER_LINE_LENGTH);
    if (e == currentPtr) strncat(str, "(CP)", PVT_DEBUG_BUFFER_LINE_LENGTH);
    if (e == backPtr) strncat(str, "(BP)", PVT_DEBUG_BUFFER_LINE_LENGTH);
    strncat(str, " ", PVT_DEBUG_BUFFER_LINE_LENGTH);
    e = e->next;
  }
  char *eqs=eqh->dumpString();
  strncat(str, eqs, PVT_DEBUG_BUFFER_LINE_LENGTH);
  delete [] tempStr;
  delete [] eqs;
  strncat(str, "end EVQ]", PVT_DEBUG_BUFFER_LINE_LENGTH);
  return str;
}

/// Pack/unpack/sizing operator
void eventQueue::pup(PUP::er &p) 
{
  p|tsOfLastInserted; p|recentAvgEventSparsity;
  p|sparsityStartTime; p|sparsityCalcCount;
  Event *tmp;
  register int i;
  int countlist = 0;
  if (p.isUnpacking()) { // UNPACKING
    p(countlist); // unpack count of events in list
    tmp = frontPtr; // front & backPtrs should already exist
    for (i=0; i<countlist; i++) { // unpack countlist events
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
    for (i=0; i<countlist; i++) { // pack each event
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
    CmiAssert((tmp->next == backPtr) || 
	      (tmp->timestamp <= tmp->next->timestamp));
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
#ifdef MEM_TEMPORAL
    CmiAssert(frontPtr->next->serialCPdata);
#else
    CmiAssert(frontPtr->next->cpData);
#endif

  // Rollback check
  tmp = frontPtr->next;
  while ((tmp != currentPtr) && (tmp->done == 1))
    tmp = tmp->next;
  if (tmp == currentPtr) CmiAssert(RBevent == NULL);
  else CmiAssert((RBevent == NULL) || (tmp == RBevent));

  // check eqheap
  eqh->sanitize();
}
