/// Sim is the base class for all poser entities
#include "pose.h"
#include "sim.def.h"

/// Global readonly proxy to array containing all posers in a simulation
CProxy_sim POSE_Objects;
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
  localPVT = (PVT *)CkLocalBranch(ThePVT);
#ifdef LB_ON
  localLBG = TheLBG.ckLocalBranch();
#endif
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  active = DOs = UNDOs = 0;
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
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (!tstat)  localStats->TimerStart(SIM_TIMER);
  else localStats->SwitchTimer(SIM_TIMER);
#endif
  prioMsg *pm;
  switch (myStrat->STRAT_T) { // step based on strategy type
  case CONS_T:
  case OPT_T:
  case OPT2_T: // pass this step call directly to strategy
    myStrat->Step();
    break;
  case OPT3_T: // prioritize this step call if work exists
    if (eq->currentPtr->timestamp > -1) {
      pm = new prioMsg;
      pm->setPriority(eq->currentPtr->timestamp-INT_MAX);
      POSE_Objects[thisIndex].Step(pm);
    }
    break;
  case SPEC_T:
  case ADAPT_T:
  case ADAPT2_T: // pass this step call directly to strategy
    myStrat->Step();
    break;
  default: 
    CkPrintf("Invalid strategy type: %d\n", myStrat->STRAT_T); 
    break;
  }
#ifdef POSE_STATS_ON
  if (!tstat)  localStats->TimerStop();
  else localStats->SwitchTimer(tstat);
#endif
}

/// Start a prioritized forward execution step on myStrat
void sim::Step(prioMsg *m)
{
  CkFreeMsg(m);
  if (active < 0) return; // object is migrating; deactivate it 
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (!tstat)
    localStats->TimerStart(SIM_TIMER);
  else localStats->SwitchTimer(SIM_TIMER);
#endif
  myStrat->Step(); // Call Step on strategy
#ifdef POSE_STATS_ON
  if (!tstat)
    localStats->TimerStop();
  else localStats->SwitchTimer(tstat);
#endif
}

/// Commit events based on new GVT estimate
void sim::Commit()
{
  static int lastGVT = 0;
  if (active < 0)  return; // object is migrating
#ifdef POSE_STATS_ON
    int tstat = localStats->TimerRunning();
    if (!tstat)  localStats->TimerStart(SIM_TIMER);
    else localStats->SwitchTimer(SIM_TIMER);
#endif
  localPVT = (PVT *)CkLocalBranch(ThePVT);
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(MISC_TIMER);
#endif
  if (localPVT->getGVT() != lastGVT) {
    if (localPVT->done() && (POSE_endtime == -1)) { // simulation inactive
      eq->CommitEvents(this, -1); // commit all events in queue
      objID->terminus(); // call terminus on all posers
    }
    else { 
      eq->CommitEvents(this, localPVT->getGVT()); // commit events up to GVT
      if (localPVT->done()) objID->terminus(); // if sim done, term posers
    }
  }
  lastGVT = localPVT->getGVT();
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(SIM_TIMER);
#endif
  if (!localPVT->done()) Step(); // not done; try stepping again
#ifdef POSE_STATS_ON
  if (!tstat)  localStats->TimerStop();
  else localStats->SwitchTimer(tstat);
#endif
}

/// Report load information to local load balancer
void sim::ReportLBdata()
{
#ifdef LB_ON
  double rbOh;
  int numEvents = 0;
  Event *tmp = eq->currentPtr;

  if (DOs-UNDOs == 0) rbOh = 1.0;
  else rbOh = ((double)DOs)/((double)(DOs-UNDOs));
  while (tmp->timestamp > -1) {
    numEvents++;
    tmp = tmp->next;
  }
  localLBG->objUpdate(myLBidx, objID->ovt, eq->currentPtr->timestamp,
		      numEvents, rbOh, srVector);
  DOs = UNDOs = 0;
  for (int i=0; i<CkNumPes(); i++) srVector[i] = 0;
#endif
}

/// Add m to cancellation list
void sim::Cancel(cancelMsg *m) 
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(CAN_TIMER);
#endif
  localPVT = (PVT *)CkLocalBranch(ThePVT);
  cancels.Insert(m->timestamp, m->evID); // add to cancellations list
  localPVT->objUpdate(m->timestamp, RECV); // tell PVT branch about recv
  CkFreeMsg(m);
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(SIM_TIMER);      
#endif
  myStrat->Step(); // call Step to handle cancellation
#ifdef POSE_STATS_ON
  localStats->TimerStop();
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

