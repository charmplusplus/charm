/// Basic Optimistic Synchronization Strategy
/** Performs locally available events in strict timestamp order */
#ifndef OPT_H
#define OPT_H

class opt : public strat {
protected:
  /// Idle measure
  int idle;
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
  virtual void CancelUnexecutedEvents();          
  /// Undo a single event, cancelling its spawned events
  virtual void UndoEvent(Event *e);     
public:
  /// Checkpoint rate
  /** Checkpoint once for every cpRate events */
  int cpRate;
  // Set of variables for monitoring object behavior
  unsigned int specEventCount, eventCount, stepCount, avgEventsPerStep;
  unsigned int rbCount, jumpCount;
  POSE_TimeType avgRBoffset, avgTimeLeash, avgJump;
  unsigned short int rbFlag;
  /// Basic Constructor
  opt() :   specEventCount(0),
    eventCount(0), stepCount(0), avgEventsPerStep(0), rbCount(0), jumpCount(0),
    rbFlag(0), idle(0), avgTimeLeash(0), avgJump(0)
    { 
      STRAT_T=OPT_T;
      cpRate=pose_config.store_rate;
      avgRBoffset = POSE_TimeMax/2;
    }
  /// Initialize the synchronization strategy type of the poser
  inline void initSync() { parent->sync = OPTIMISTIC; }
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
    if ((ec == POSE_UnsetTS) && (worktime == POSE_UnsetTS)) {
      return POSE_UnsetTS;
    }
    if (ec > POSE_UnsetTS) {
      theTime = ec;
      if ((worktime > POSE_UnsetTS) && (worktime < ec) && (ovt < worktime))
	theTime = worktime;
      else if ((worktime > POSE_UnsetTS) && (worktime < ec) && (ovt < ec))
	theTime = ovt;
      else if ((worktime > POSE_UnsetTS) && (worktime < ec))
	theTime = ec;
      else if ((worktime == POSE_UnsetTS) && (ovt < ec))
	theTime = ec;
    }
    else if (worktime > POSE_UnsetTS) {
      theTime = worktime;
      if (ovt > worktime)
	theTime = ovt;
    }
    //if (theTime == 15999)
    //CkPrintf("theTime=%d ovt=%d wt=%d ec=%d gvt=%d\n", theTime, ovt, 
    //worktime, ec, gvt);
    CkAssert((theTime == POSE_UnsetTS) || (theTime >= gvt) ||
	     (theTime == (gvt-1)));
    return theTime;
  }
  /// Add spawned event to current event's spawned event list
  inline void AddSpawnedEvent(int AnObjIdx, eventID evID, POSE_TimeType ts) { 
    eq->AddSpawnToCurrent(AnObjIdx, evID, ts);
  }
  /// Send cancellation messages to all of event e's spawned events
  void CancelSpawn(Event *e) {  
#if !CMK_TRACE_DISABLED
    double critStart;
    if(pose_config.trace)
      critStart= CmiWallTimer();
#endif
    cancelMsg *m;
    SpawnedEvent *ev = e->spawnedList;
    while (ev) {
      e->spawnedList = ev->next; // remove a spawn from the list
      ev->next = NULL;
      m = new cancelMsg(); // build a cancel message
      m->evID = ev->evID;
      m->timestamp = ev->timestamp;
      m->setPriority((m->timestamp-1) - POSE_TimeMax);
      localPVT->objUpdate(ev->timestamp, SEND);
      //char str[20];
      //CkPrintf("[%d] SEND(cancel) %s at %d...\n", CkMyPe(), ev->evID.sdump(str), ev->timestamp);      
      POSE_Objects[ev->objIdx].Cancel(m); // send the cancellation
      delete ev; // delete the spawn
      ev = e->spawnedList; // move on to next in list
    }
#if !CMK_TRACE_DISABLED
    if(pose_config.trace)
      traceUserBracketEvent(30, critStart, CmiWallTimer());
#endif
  }
};

#endif
