/// Sim is the base class for all poser entities
#include "pose.h"
#include "sim.def.h"

/// Global readonly proxy to array containing all posers in a simulation
CProxy_sim POSE_Objects;
CProxy_sim POSE_Objects_RO;
/// Coordinates all startup and shutdown behaviors for POSE simulations
CkChareID POSE_Coordinator_ID;

#ifdef POSE_COMM_ON
/// Used with the CommLib
extern CkGroupID dmid;
//extern int com_debug;
#endif

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
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
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
  ArrayElement1D::pup(p); // call parent class pup method
  // pup simple types
  p(active); p(myPVTidx); p(myLBidx); p(sync); p(DOs); p(UNDOs);
  // pup event queue
  if (p.isUnpacking())
    eq = new eventQueue();
  eq->pup(p);
  // pup cancellations
  cancels.pup(p);
  if (p.isUnpacking()) { // reactivate migrated object
#ifndef CMK_OPTIMIZE
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
  // pup checkpoint info for sequential mode using sim 0 only
#ifdef SEQUENTIAL_POSE
  if (thisIndex == 0) {
    p|seqCheckpointInProgress;
    p|seqLastCheckpointGVT;
    p|seqLastCheckpointTime;
    p|seqStartTime;
    p|POSE_Skipped_Events;
    if (p.isUnpacking()) {
      seqStartTime = CmiWallTimer() - (seqLastCheckpointTime - seqStartTime);
    }
  }
#endif
}

/// Start a forward execution step on myStrat
void sim::Step()
{
  if (active < 0) return; // object is migrating; deactivate it 

#ifndef CMK_OPTIMIZE
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
  case ADAPT4_T: // pass this step call directly to strategy
    myStrat->Step();
    break;
  default: 
    CkPrintf("Invalid strategy type: %d\n", myStrat->STRAT_T); 
    break;
  }
#ifndef CMK_OPTIMIZE
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
#ifndef CMK_OPTIMIZE
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

#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    {
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
#ifndef CMK_OPTIMIZE
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
#ifndef CMK_OPTIMIZE
  if(pose_config.trace) {
    traceUserBracketEvent(50, critStart, CmiWallTimer());
    critStart = CmiWallTimer();
  }
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);
#endif
  if (!isDone && (eq->currentPtr->timestamp > -1)) 
    Step(); // not done; try stepping again

#ifndef CMK_OPTIMIZE
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
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(CAN_TIMER);
#endif
  //  char str[20];
  //  CkPrintf("[%d] RECV(cancel) %s at %d...\n", CkMyPe(), m->evID.sdump(str), m->timestamp);      
  //localPVT = (PVT *)CkLocalBranch(ThePVT);
  cancels.Insert(m->timestamp, m->evID); // add to cancellations list
  localPVT->objUpdate(m->timestamp, RECV); // tell PVT branch about recv
  CkFreeMsg(m);

#ifndef CMK_OPTIMIZE
  double critStart;
  if(pose_config.trace)
    critStart= CmiWallTimer();  // trace timing
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);      
#endif

  myStrat->Step(); // call Step to handle cancellation

#ifndef CMK_OPTIMIZE
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
  CkPrintf("POSE: beginning checkpoint on sim %d at GVT=%lld time=%.1f sec\n", thisIndex, seqLastCheckpointGVT, CmiWallTimer() - seqStartTime);
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
  CkPrintf("POSE: checkpoint/restart complete on sim %d at GVT=%lld time=%.1f sec\n", thisIndex, POSE_GlobalClock, CmiWallTimer() - seqStartTime);
  // restart simulation
  while (POSE_Skipped_Events.length() > 0) {
    // These Step iterations MUST be executed now, before any messages
    // are delivered, or else the event queues will break.  To do this
    // efficiently, since we're in sequential mode, call Step() as a
    // local function.
    int index = POSE_Skipped_Events.deq();
    sim *localSim = POSE_Objects[index].ckLocal();
    if (localSim == NULL) {
      CkPrintf("ERROR: could not obtain pointer to local sim object %d after checkpoint/restart\n", index);
      CkAbort("Pointer to local sim is NULL...this shouldn't happen in sequential mode\n");
    } else {
      localSim->Step();
    }
  }
  CkStartQD(CkIndex_pose::stop(), &POSE_Coordinator_ID);
}

void sim::ResumeFromSync()
{
  PVT *localPVT = (PVT *)CkLocalBranch(ThePVT);
  localPVT->doneLB();
}

/// Dump all data fields
void sim::dump()
{
  CkPrintf("[SIM: active=%d sync=%d myPVTidx=%d ", active, sync, myPVTidx);
  if (objID) objID->dump();
  else CkPrintf("objID=NULL\n");
  eq->dump();
  cancels.dump();
  CkPrintf("end SIM]\n");
}

