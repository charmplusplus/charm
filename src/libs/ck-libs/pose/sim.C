//  File: sim.C
//  For header info, see sim.h.
//  The sim base class implements the wrapper around the user's simulation
//  classes.  It holds the event queue, the actual simulated object (rep), 
//  a strategy, and a list of cancel messages. 
//  Last modified: 07.31.01 by Terry Wilmarth

#include "pose.h"
#include "sim.def.h"

CProxy_sim POSE_Objects;
CkChareID POSE_Coordinator_ID;

#ifdef POSE_COMM_ON
extern CkGroupID dmid;
//extern int comm_debug;
#endif

sim::sim() // sim base constructor
{
  localPVT = (PVT *)CkLocalBranch(ThePVT);
#ifdef LB_ON
  localLBG = TheLBG.ckLocalBranch();
#endif
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  recycCount = active = DOs = UNDOs = 0;
  srVector = (int *)malloc(CkNumPes() * sizeof(int));
  for (int i=0; i<CkNumPes(); i++) srVector[i] = 0;
  eq = new eventQueue();
  myStrat = NULL;
  objID = NULL;
}

sim::~sim() // basic destructor
{
  delete(eq);
  delete(myStrat);
  delete(objID);
  for (int i=0; i<recycCount; i++)
    delete(recyc[i]);
}

// Creates a forward execution step with myStrat
void sim::Step()
{
  if (active < 0) return;  // object is migrating; deactivate it 
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (!tstat)
    localStats->TimerStart(SIM_TIMER);
  else localStats->SwitchTimer(SIM_TIMER);
#endif
  prioMsg *pm = new prioMsg;
  switch (myStrat->STRAT_T) {
  case CONS_T:
  case OPT_T:
  case OPT2_T:
  case OPT3_T: // prioritized step
    if (eq->currentPtr->timestamp >= 0) {
      pm->setPriority(eq->currentPtr->timestamp-INT_MAX);
      POSE_Objects[thisIndex].Step(pm);
    }
    else myStrat->Step();
    break;
  case SPEC_T:
  case ADAPT_T: 
  case ADAPT2_T: // non-prioritized step
    myStrat->Step(); // Call Step on strategy
    break;
  default: 
    CkPrintf("Invalid strategy type: %d\n", myStrat->STRAT_T); 
    break;
  }
#ifdef POSE_STATS_ON
  if (!tstat)
    localStats->TimerStop();
  else localStats->SwitchTimer(tstat);
#endif
}

void sim::Step(prioMsg *m)
{
  CkFreeMsg(m);
  if (active < 0) return;  // object is migrating; deactivate it 
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

void sim::Status()
{
  //int st = myStrat->SafeTime();
  //  if (st == localPVT->getGVT())
  //    CkPrintf("Object %d has safeTime == GVT...\n", thisIndex);
  //localPVT->objUpdate(myPVTidx, st, -1, -1);
  localPVT->objUpdate(myPVTidx, myStrat->SafeTime(), -1, -1);
}

// Commit whatever can be committed according to new GVT value
void sim::Commit()
{
  if (active < 0)
    return;
  //#ifdef POSE_STATS_ON
  //  int tstat = localStats->TimerRunning();
  //  if (!tstat)
  //    localStats->TimerStart(MISC_TIMER);
  //  else localStats->SwitchTimer(MISC_TIMER);
  //#endif

#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (!tstat)
    localStats->TimerStart(SIM_TIMER);
  else localStats->SwitchTimer(SIM_TIMER);
#endif

  if (localPVT->done() && (POSE_endtime == -1)) { //commit all events in queue
    eq->CommitEvents(this, -1);
    objID->terminus();
  }
  else {
    eq->CommitEvents(this, localPVT->getGVT());
    if (localPVT->done())
      objID->terminus();
  }
  objID->ResetCheckpointRate();
  //#ifdef POSE_STATS_ON
  //  localStats->SwitchTimer(SIM_TIMER);
  //#endif
  if (!localPVT->done() && 
      ((eq->currentPtr->timestamp >= 0) || !cancels.IsEmpty())) {
    Step();
  }
#ifdef POSE_STATS_ON
  if (!tstat)
    localStats->TimerStop();
  else localStats->SwitchTimer(tstat);
#endif
}

void sim::ReportLBdata()
{
#ifdef LB_ON
  double rbOh;
  int numEvents = 0;
  Event *tmp = eq->currentPtr;

  if (DOs-UNDOs == 0) rbOh = 1.0;
  else rbOh = ((double)DOs)/((double)(DOs-UNDOs));
  while (tmp->timestamp >= 0) {
    numEvents++;
    tmp = tmp->next;
  }
  localLBG->objUpdate(myLBidx, objID->ovt, eq->currentPtr->timestamp,
		      numEvents, rbOh, srVector);
  DOs = UNDOs = 0;
  for (int i=0; i<CkNumPes(); i++) srVector[i] = 0;
#endif
}

void sim::Migrate(destMsg *m)
{
  migrateMe(m->destPE);
}

// Overidden by user code
void sim::ResolveFn(int fnid, void *msg) { }
void sim::ResolveCommitFn(int fnid, void *msg) { }

// Add m to cancellation list
void sim::Cancel(cancelMsg *m) 
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(CAN_TIMER);
#endif
  cancels.Insert(m->timestamp, m->evID);    // add to cancellations list
  localPVT->objUpdate(m->timestamp, RECV);
  if (eq->currentPtr->timestamp == -1)  // if no work, Step to handle cancel
    Step();
  CkFreeMsg(m);
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

// dump the entire sim object
void sim::dump(int pdb_level)
{
  pdb_indent(pdb_level);
  CkPrintf("[SIM: active=%d sync=%d myPVTidx=%d ", active, sync, myPVTidx);
  if (objID) objID->dump(pdb_level+1);
  else CkPrintf("objID=NULL\n");
  eq->dump(pdb_level+1);
  cancels.dump(pdb_level+1);
  pdb_indent(pdb_level);
  CkPrintf("end SIM]\n");
}

