/// Basic Optimistic Synchronization Strategy
/** Performs locally available events in strict timestamp order */
#ifndef OPT_H
#define OPT_H

class opt : public strat {
protected:
  /// Rollback to predetermined RBevent
  virtual void Rollback();              
  /// Recover checkpointed state prior to ev
  /** Searches backward from recoveryPoint without undoing events or
      cancelling their spawn, until a checkpoint is found. Then
      restores the state from this found checkpoint, and uses
      pseudo-re-execution to return back to recoveryPoint.  This means
      that events are re-executed to reconstruct the state, but any
      events they would have spawned are not respawned, since they
      were never cancelled. */
  virtual void RecoverState(Event *recoveryPoint); 
  /// Cancel events in cancellation list that have arrived
  /** This will go through all the cancellations and remove whatever has 
      arrived already.  For events that have already been executed, a 
      rollback is performed.  No forward execution happens until all the
      cancellations have been examined. */
  virtual void CancelEvents();          
  /// Undo a single event, cancelling its spawned events
  virtual void UndoEvent(Event *e);     
public:
  /// Checkpoint rate
  /** Checkpoint once for every cpRate events */
  int cpRate;
  /// Basic Constructor
  opt() { STRAT_T = OPT_T; cpRate = STORE_RATE; }
  /// Initialize the synchronization strategy type of the poser
  void initSync() { parent->sync = OPTIMISTIC; }
  /// Perform a single forward execution step
  /** Prior to the forward execution, cancellation and rollback are done if
      necessary.  Derived strategies typically just reimplement this method */
  virtual void Step();              
  /// Compute safe time for object
  /** Safe time is the earliest timestamp that this object can generate given
      its current state (assuming no stragglers, cancellations or events
      are subsequently received */
  POSE_TimeType SafeTime() {  
    POSE_TimeType ovt=userObj->OVT(), theTime=POSE_UnsetTS,
      ec=parent->cancels.getEarliest(), gvt=localPVT->getGVT(), 
      worktime = eq->currentPtr->timestamp;
    // Object is idle; report -1
    if (!RBevent && (ec == POSE_UnsetTS) && (worktime == POSE_UnsetTS) && 
	(ovt <= gvt))  
      return POSE_UnsetTS;
    if (RBevent)  theTime = RBevent->timestamp;
    if ((ec > POSE_UnsetTS) && ((ec < theTime) || (theTime == POSE_UnsetTS)))  
      theTime = ec;
    POSE_TimeType maxWork = worktime;
    if (ovt > maxWork) maxWork = ovt;
    if ((worktime == POSE_UnsetTS) && (ovt <= gvt)) maxWork = POSE_UnsetTS;
    if (((maxWork != POSE_UnsetTS) && (maxWork < theTime)) || 
	(theTime == POSE_UnsetTS))
      theTime = maxWork;
    CkAssert((theTime == POSE_UnsetTS) || (theTime >= gvt));
    //if (theTime == gvt) CkPrintf("safe=%d ovt=%d wt=%d ec=%d gvt=%d\n", theTime, ovt, worktime, ec, gvt);
    return theTime;
  }
  /// Add spawned event to current event's spawned event list
  void AddSpawnedEvent(int AnObjIdx, eventID evID, POSE_TimeType ts) { 
    eq->AddSpawnToCurrent(AnObjIdx, evID, ts);
  }
  /// Send cancellation messages to all of event e's spawned events
  void CancelSpawn(Event *e) {  
    cancelMsg *m;
    SpawnedEvent *ev = e->spawnedList;
    while (ev) {
      e->spawnedList = ev->next; // remove a spawn from the list
      ev->next = NULL;
      m = new cancelMsg(); // build a cancel message
      m->evID = ev->evID;
      m->timestamp = ev->timestamp;
      m->setPriority(m->timestamp - POSE_TimeMax);
      localPVT->objUpdate(ev->timestamp, SEND);
      //CkPrintf("Cancelling spawned event "); ev->evID.dump(); CkPrintf("\n");
      POSE_Objects[ev->objIdx].Cancel(m); // send the cancellation
      delete ev; // delete the spawn
      ev = e->spawnedList; // move on to next in list
    }
  }
};

#endif
