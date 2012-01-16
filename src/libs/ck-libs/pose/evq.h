/// Queue of executed and unexecuted events on a poser
#ifndef EVQ_H
#define EVQ_H

//#define EQ_SANITIZE 1

/// The event queue
/** Doubly-linked list with front and back sentinels and a heap of unexecuted
    events */
class eventQueue {
  /// This helper method cleans up all the commit code by removing the stats
  void CommitStatsHelper(sim *obj, Event *commitPtr);
 public:
  /// Sentinel nodes
  Event *frontPtr, *backPtr;
  /// Pointer to next unexecuted event
  Event *currentPtr;
  /// Event to rollback to
  Event *RBevent;
  /// Heap of unexecuted events
  EqHeap *eqh;
  /// Largest unexecuted event timestamp in queue
  POSE_TimeType largest;
  /// number of unexecuted events in the queue
  unsigned int eventCount;
  /// Timestamp of the last event inserted in the queue
  POSE_TimeType tsOfLastInserted;
  /// Coarse memory usage
  unsigned int mem_usage;
  /// Keep track of last logged VT for this object so no duplicates are logged
  POSE_TimeType lastLoggedVT;
  /// Average sparsity of recently committed events (in GVT ticks / event)
  int recentAvgEventSparsity;
  /// Timestamp of the first event for the sparsity calculation
  POSE_TimeType sparsityStartTime;
  /// Sparsity calculation counter
  int sparsityCalcCount;
  /// Counts the differences examined; used for timeleash calculation in adapt5
  int tsDiffCount;
  /// The timestamp of the last committed event
  POSE_TimeType lastCommittedTS;
  /// The largest timestamp differences at commit time; used for timeleash calculation in adapt5
  POSE_TimeType tsCommitDiffs[DIFFS_TO_STORE];
#ifdef MEM_TEMPORAL
  TimePool *localTimePool;
#endif
  /// Basic Constructor
  /** Creates front and back sentinel nodes, connects them, and inits pointers
      and heap. */
  eventQueue();
  /// Destructor
  ~eventQueue();
  /// Insert e in the queue in timestamp order
  /** If executed events with same timestamp exist, insert e at the back of 
      these, returns 0 if rollback necessary, 1 otherwise. */
  void InsertEvent(Event *e);      
  /// Insert e in the queue in timestamp order
  /** If executed events with same timestamp exist, sort e into them based on
      eventID, returns 0 if rollback necessary, 1 otherwise. */
  void InsertEventDeterministic(Event *e);      
  /// Return front pointer
  inline Event *front() { return frontPtr; }
  /// Return back pointer
  inline Event *back() { return backPtr; }
  /// Move currentPtr to next event in queue
  /** If no more events, take one from heap */
  void ShiftEvent() { 
    Event *e;
#ifdef EQ_SANITIZE
    sanitize();
#endif
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
    else FindLargest();
    eventCount--;
#ifdef EQ_SANITIZE
    sanitize();
#endif
  }
  /// Commit (delete) events before target timestamp ts
  void CommitEvents(sim *obj, POSE_TimeType ts); 
  /// Commit (delete) all events
  void CommitAll(sim *obj); 
  /// Commit (delete) all events that are done (used in sequential mode)
  void CommitDoneEvents(sim *obj);
  /// Change currentPtr to point to event e
  /** Be very very careful with this -- avoid using if possible */
  void SetCurrentPtr(Event *e);
  /// Delete event and reconnect surrounding events in queue
  void DeleteEvent(Event *ev);
  /// Return the event ID of the event pointed to by currentPtr
  inline const eventID& CurrentEventID() { return currentPtr->evID; }
  /// Add id, e and ts as an entry in currentPtr's spawned list
  /** The poser e was sent to is id */
  inline void AddSpawnToCurrent(int id, eventID e, POSE_TimeType ts) {
    SpawnedEvent *newnode = 
      new SpawnedEvent(id, e, ts, currentPtr->spawnedList);
    CmiAssert(currentPtr->done == 2);
    currentPtr->spawnedList = newnode;
  }
  /// Return the first entry in currentPtr's spawned list and remove it
  inline SpawnedEvent *GetNextCurrentSpawn() {
    SpawnedEvent *tmp = currentPtr->spawnedList;
    if (tmp) currentPtr->spawnedList = tmp->next;
    return tmp;
  }
  /// Find the largest timestamp of the unexecuted events
  inline void FindLargest() {
    POSE_TimeType hs = eqh->FindMax();
    if (backPtr->prev->done == 0) largest = backPtr->prev->timestamp;
    else largest = POSE_UnsetTS;
    if (largest < hs) largest = hs;
  }
  /// Set rollback point to event e
  void SetRBevent(Event *e) {
    if (!RBevent) RBevent = e; 
    else if ((pose_config.deterministic) && (RBevent->timestamp > e->timestamp) ||
	     ((RBevent->timestamp == e->timestamp) && 
	      (e->evID < RBevent->evID))) {
      CmiAssert(RBevent->prev->next == RBevent);
      CmiAssert(RBevent->next->prev == RBevent);
      RBevent = e;
    }
    else if ((RBevent->timestamp > e->timestamp) ||
             (RBevent->timestamp == e->timestamp && RBevent->evID > e->evID)
            ) {
      CmiAssert(RBevent->prev->next == RBevent);
      CmiAssert(RBevent->next->prev == RBevent);
      RBevent = e;
    }
  }
  /// Dump the event queue
  void dump();
  /// Dump the event queue to a string
  char *dumpString();
  /// Pack/unpack/sizing operator
  void pup(PUP::er &p);   
  /// Check validity of data fields
  void sanitize();
};

#endif
