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
//extern int comm_debug;
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
  delete(eq);
  delete(myStrat);
  delete(objID);
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
#ifdef MEM_TEMPORAL
  else {  // What's the constant doing to us?
    lastGVT = curGVT;
    eq->CommitEvents(this, lastGVT); // commit events up to GVT
  }
#else
  else if (curGVT > lastGVT + 100) {  // What's the constant doing to us?
    lastGVT = curGVT;
    eq->CommitEvents(this, lastGVT); // commit events up to GVT
  }
#endif
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

