/// Sim is the base class for all poser entities
#include "pose.h"
#include "poseMsgs.def.h"
#include "sim.def.h"

/// Global readonly proxy to array containing all posers in a simulation
CProxy_sim POSE_Objects;
CProxy_sim POSE_Objects_RO;
/// Coordinates all startup and shutdown behaviors for POSE simulations
CkChareID POSE_Coordinator_ID;

/// Basic Constructor
sim::sim() 
{
#ifdef VERBOSE_DEBUG
  CkPrintf("[%d] constructing sim %d\n",CkMyPe(), thisIndex);
#endif

#ifndef SEQUENTIAL_POSE
  localPVT = (PVT *)CkLocalBranch(ThePVT);
  if(pose_config.lb_on)
    localLBG = TheLBG.ckLocalBranch();
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  basicStats[0] = basicStats[1] = 0LL;
  lastGVT = active = DOs = UNDOs = 0;
  srVector = (int *)malloc(CkNumPes() * sizeof(int));
  for (int i=0; i<CkNumPes(); i++) srVector[i] = 0;
  eq = new eventQueue();
  myStrat = NULL;
  objID = NULL;
}

/// Destructor
sim::~sim() 
{
  active = -1;
#ifndef SEQUENTIAL_POSE
  localPVT->objRemove(myPVTidx);
#endif
  if(pose_config.lb_on)
    localLBG->objRemove(myLBidx);

  delete(eq);
  delete(myStrat);
  delete(objID);
}

/// Pack/unpack/sizing operator
void sim::pup(PUP::er &p) {
  // pup simple types
  p(active); p(myPVTidx); p(myLBidx); p(sync); p(DOs); p(UNDOs);
  // pup event queue
  if (p.isUnpacking()) {
    eq = new eventQueue();
  }
  eq->pup(p);
  // pup cancellations
  cancels.pup(p);
  if (p.isUnpacking()) { // reactivate migrated object
#if !CMK_TRACE_DISABLED
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
#ifndef SEQUENTIAL_POSE
    localPVT = (PVT *)CkLocalBranch(ThePVT);
    myPVTidx = localPVT->objRegister(thisIndex, localPVT->getGVT(), sync, this);
    if(pose_config.lb_on){
      localLBG = TheLBG.ckLocalBranch();
      myLBidx = localLBG->objRegister(thisIndex, sync, this);
    }
#endif
    active = 0;
  }
  PUParray(p, basicStats, 2);
  // pup checkpoint info for sequential mode using sim 0 only
#ifdef SEQUENTIAL_POSE
  if (thisIndex == 0) {
    p|seqCheckpointInProgress;
    p|seqLastCheckpointGVT;
    p|seqLastCheckpointTime;
    p|seqStartTime;
    p|POSE_Skipped_Events;
    p|poseIndexOfStopEvent;
    if (p.isUnpacking()) {
      seqStartTime = seqLastCheckpointTime;
    }
  }
#endif
}

/// Start a forward execution step on myStrat
void sim::Step()
{
  if (active < 0) return; // object is migrating; deactivate it 

#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart=CmiWallTimer();  // trace timing
  int tstat;
  if(pose_config.stats)
    {
      tstat= localStats->TimerRunning();
      if (!tstat)  localStats->TimerStart(SIM_TIMER);
      else localStats->SwitchTimer(SIM_TIMER);
    }
#endif

  prioMsg *pm;
  switch (myStrat->STRAT_T) { // step based on strategy type
  case SEQ_T:
  case CONS_T:
  case OPT_T:
  case OPT2_T: // pass this step call directly to strategy
  case OPT3_T: // prioritize this step call if work exists
  case SPEC_T:
  case ADAPT_T:
  case ADAPT2_T:
  case ADAPT3_T:
  case ADAPT4_T:
  case ADAPT5_T: // pass this step call directly to strategy
    myStrat->Step();
    break;
  default: 
    CkPrintf("Invalid strategy type: %d\n", myStrat->STRAT_T); 
    break;
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)  
    {
      if (!tstat)  localStats->TimerStop();
      else localStats->SwitchTimer(tstat);
    }
  if(pose_config.trace)
    traceUserBracketEvent(60, critStart, CmiWallTimer());
#endif
}

/// Start a prioritized forward execution step on myStrat
void sim::Step(prioMsg *m)
{
  CkFreeMsg(m);
  if (active < 0) return; // object is migrating; deactivate it 
#if !CMK_TRACE_DISABLED
  int tstat;
  if(pose_config.stats)
    {
      tstat= localStats->TimerRunning();
      if (!tstat)
	localStats->TimerStart(SIM_TIMER);
      else localStats->SwitchTimer(SIM_TIMER);
    }
#endif

  myStrat->Step(); // Call Step on strategy

#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    {
      if (!tstat)
	localStats->TimerStop();
      else localStats->SwitchTimer(tstat);
    }
#endif
}

/// Start a forward execution step on myStrat after a checkpoint (sequential mode only)
void sim::CheckpointStep(eventMsg *m) {
  CkFreeMsg(m);
  if (active < 0) return; // object is migrating; deactivate it 
#if !CMK_TRACE_DISABLED
  int tstat;
  if (pose_config.stats) {
    tstat = localStats->TimerRunning();
    if (!tstat)
      localStats->TimerStart(SIM_TIMER);
    else localStats->SwitchTimer(SIM_TIMER);
  }
#endif

  // ensure sequential mode
  CkAssert(myStrat->STRAT_T == SEQ_T);
  myStrat->Step(); // Call Step on strategy

#if !CMK_TRACE_DISABLED
  if (pose_config.stats) {
    if (!tstat)
      localStats->TimerStop();
    else localStats->SwitchTimer(tstat);
  }
#endif
}

/// Commit events based on new GVT estimate
void sim::Commit()
{
  if (active < 0)  return; // object is migrating
#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart= CmiWallTimer();  // trace timing
  int tstat;
  if(pose_config.stats) {
    tstat = localStats->TimerRunning();
    if (!tstat)  localStats->TimerStart(SIM_TIMER);
    else localStats->SwitchTimer(SIM_TIMER);
  }
  if(pose_config.stats)
    localStats->SwitchTimer(FC_TIMER);
#endif
  int isDone=localPVT->done(); 
  int curGVT=localPVT->getGVT();
  if (isDone) { // simulation inactive
    eq->CommitEvents(this, POSE_endtime); // commit all events in queue
    Terminate();// call terminus on all posers
  }
  else if (curGVT > lastGVT + 100) {  // What's the constant doing to us?
    lastGVT = curGVT;
    eq->CommitEvents(this, lastGVT); // commit events up to GVT
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.trace) {
    traceUserBracketEvent(50, critStart, CmiWallTimer());
    critStart = CmiWallTimer();
  }
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);
#endif
  if (!isDone && (eq->currentPtr->timestamp > -1))
    Step(); // not done; try stepping again

#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (!tstat)  localStats->TimerStop();
    else localStats->SwitchTimer(tstat);
  if(pose_config.trace)
    traceUserBracketEvent(60, critStart, CmiWallTimer());
#endif
}

/// Commit all possible events before a checkpoint to disk
/*  This is necessary to ensure a minimal state at checkpoint time.
    sim::Commit() requires a minimum advancement in GVT (currently 100
    ticks) before the event queue commits its events.  As a result,
    some events could be left in the queue.  This function takes care
    of that.
*/
void sim::CheckpointCommit() {
  if (active < 0)  return; // object is migrating
#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart= CmiWallTimer();  // trace timing
  int tstat;
  if(pose_config.stats) {
    tstat = localStats->TimerRunning();
    if (!tstat)  localStats->TimerStart(SIM_TIMER);
    else localStats->SwitchTimer(SIM_TIMER);
  }
  if(pose_config.stats)
    localStats->SwitchTimer(FC_TIMER);
#endif
  int curGVT = localPVT->getGVT();
  lastGVT = curGVT;
  eq->CommitEvents(this, lastGVT); // commit everything up to the current GVT
#if !CMK_TRACE_DISABLED
  if(pose_config.trace) {
    traceUserBracketEvent(50, critStart, CmiWallTimer());
    critStart = CmiWallTimer();
  }
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);
#endif
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    if (!tstat)  localStats->TimerStop();
    else localStats->SwitchTimer(tstat);
  if(pose_config.trace)
    traceUserBracketEvent(60, critStart, CmiWallTimer());
#endif
}

/// Report load information to local load balancer
void sim::ReportLBdata()
{
  if(pose_config.lb_on){
    double rbOh;
    int numEvents = 0;
    Event *tmp = eq->currentPtr;

    if (DOs-UNDOs == 0) rbOh = 1.0;
    else rbOh = ((double)DOs)/((double)(DOs-UNDOs));
    while (tmp->timestamp > POSE_UnsetTS) {
      numEvents++;
      tmp = tmp->next;
    }
    localLBG->objUpdate(myLBidx, objID->ovt, eq->currentPtr->timestamp,
			numEvents, rbOh, srVector);
    DOs = UNDOs = 0;
    for (int i=0; i<CkNumPes(); i++) srVector[i] = 0;
  }
}

/// Add m to cancellation list
void sim::Cancel(cancelMsg *m) 
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStart(CAN_TIMER);
#endif
  //  char str[20];
  //  CkPrintf("[%d] RECV(cancel) %s at %d...\n", CkMyPe(), m->evID.sdump(str), m->timestamp);      
  //localPVT = (PVT *)CkLocalBranch(ThePVT);
  cancels.Insert(m->timestamp, m->evID); // add to cancellations list
  localPVT->objUpdate(m->timestamp, RECV); // tell PVT branch about recv
  CkFreeMsg(m);

#if !CMK_TRACE_DISABLED
  double critStart;
  if(pose_config.trace)
    critStart= CmiWallTimer();  // trace timing
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);      
#endif

  myStrat->Step(); // call Step to handle cancellation

#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStop();
  if(pose_config.trace)
    traceUserBracketEvent(60, critStart, CmiWallTimer());
#endif
}

// Sequential checkpointing: Two functions, SeqBeginCheckpoint and
// SeqResumeAfterCheckpoint, were added to the sim class to
// handle this.  Only initiate the checkpointing procedure on sim
// 0, after commits have occurred.  This should minimize the
// amount of data written to disk.  In order to ensure a stable
// state, we wait for quiescence to be reached before beginning
// the checkpoint.  Once this happens, sim 0 checkpoints and then
// resumes the simulation in SeqResumeAfterCheckpoint.  This
// Callback function is also the first POSE function to be called
// when restarting from a checkpoint.

// While waiting for quiescence to be reached, all events with
// timestamps less than the checkpoint GVT are allowed to
// execute.  All others are skipped, instead storing their sim handles (indices
// into the POSE_Objects array) in POSE_Skipped_Events.  Even
// though execution is skipped, the events still remain in their
// event queues.  When resuming the simulation, both after
// checkpointing and after a restart, sim::Step() (which calls
// seq::Step()) is called on each poser listed in
// POSE_Skipped_Events to execute the skipped events.

// Checkpoints are initiated approximately every
// pose_config.checkpoint_gvt_interval GVT ticks or
// pose_config.checkpoint_time_interval seconds (both defined in
// pose_config.h).

/// In sequential mode, begin checkpoint after reaching quiescence
void sim::SeqBeginCheckpoint() {
  // Ensure this only happens on sim 0
  CkAssert(thisIndex == 0);
  // Ensure we're checkpointing
  CkAssert(seqCheckpointInProgress);
  CkPrintf("POSE: quiescence detected\n");
  CkPrintf("POSE: beginning checkpoint on sim %d at GVT=%lld sim time=%.1f sec\n", thisIndex, seqLastCheckpointGVT, CmiWallTimer() + seqStartTime);
  CkCallback cb(CkIndex_sim::SeqResumeAfterCheckpoint(), CkArrayIndex1D(thisIndex), thisProxy);
  CkStartCheckpoint(POSE_CHECKPOINT_DIRECTORY, cb);
}

/// In sequential mode, resume after checkpointing or restarting
void sim::SeqResumeAfterCheckpoint() {
  // Ensure this only happens on sim 0
  CkAssert(thisIndex == 0);
  // Ensure this function is only called once after a checkpoint
  CkAssert(seqCheckpointInProgress);
  seqCheckpointInProgress = 0;
  POSE_GlobalClock = seqLastCheckpointGVT;
  CkPrintf("POSE: checkpoint/restart complete on sim %d at GVT=%lld sim time=%.1f sec\n", thisIndex, POSE_GlobalClock, CmiWallTimer() + seqStartTime);

  // restart simulation
  while (POSE_Skipped_Events.length() > 0) {
    // These Step iterations must all be enqueued now, before any messages
    // are delivered, or else the event queues will break.  Because
    // messages spawned by these events may need to be inserted, these
    // events should be enqueued in the same manner as that used by
    // the translated verion of POSE_invoke
    eventMsg *evtMsg = new eventMsg;
    Skipped_Event se = POSE_Skipped_Events.deq();
    int _POSE_handle = se.simIndex;
    POSE_TimeType _POSE_timeOffset = se.timestamp - objID->ovt;
    objID->registerTimestamp(_POSE_handle, evtMsg, _POSE_timeOffset);
    if (pose_config.dop) {
      ct = CmiWallTimer();
      evtMsg->rst = ct - st + eq->currentPtr->srt;
    }
#if !CMK_TRACE_DISABLED
    evtMsg->sanitize();
#endif
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->SwitchTimer(COMM_TIMER);
#endif
    thisProxy[_POSE_handle].CheckpointStep(evtMsg);
#if !CMK_TRACE_DISABLED
    if (pose_config.stats)
      localStats->SwitchTimer(DO_TIMER);
#endif
  }

  // restart quiescence detection, which is used for termination of
  // sequential POSE
  CkStartQD(CkIndex_pose::stop(), &POSE_Coordinator_ID);
}

void sim::ResumeFromSync()
{
//  PVT *localPVT = (PVT *)CkLocalBranch(ThePVT);
//  localPVT->doneLB();
}

/// Dump all data fields
void sim::dump()
{
  CkPrintf("[SIM: thisIndex=%d active=%d sync=%d myPVTidx=%d ", thisIndex, active, sync, myPVTidx);
  if (objID) objID->dump();
  else CkPrintf("objID=NULL\n");
  eq->dump();
  cancels.dump();
  CkPrintf("end SIM]\n");
}

