/// Global Virtual Time estimation for POSE
#include "pose.h"
#include "srtable.h"
#include "gvt.def.h"
#include "qd.h"

CkGroupID ThePVT;
CkGroupID TheGVT;

/// Basic Constructor
PVT::PVT() 
{
#ifdef POSE_COMM_ON
  //comm_debug = 1;
#endif
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
  localStats->TimerStart(GVT_TIMER);
#endif
  optPVT = conPVT = estGVT = -1;
  waitingForGVT = simdone = 0;
  SendsAndRecvs = new SRtable();
  LastSendsAndRecvs = new SRtable();
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

/// ENTRY: runs the PVT calculation and reports to GVT
void PVT::startPhase() 
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  CProxy_PVT p(ThePVT);
  CProxy_GVT g(TheGVT);
  static int gvtTurn = 0;
  UpdateMsg *umsg;
  int i;

  if (waitingForGVT) { // haven't received previous GVT estimate
    p[CkMyPe()].startPhase(); // start this later
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
    return;
  }

  objs.Wake(); // wake objects to make sure all have reported

  // compute PVT
  optPVT = conPVT = -1;
  for (i=0; i<objs.getNumSpaces(); i++)
    if (objs.objs[i].isPresent()) {
      if (objs.objs[i].isOptimistic()) { // check optPVT 
	if ((optPVT < 0) || ((objs.objs[i].getOVT() < optPVT) && 
			     (objs.objs[i].getOVT() >= 0))) 
	  optPVT = objs.objs[i].getOVT();
      }
      else if (objs.objs[i].isConservative()) { // check conPVT
	if ((conPVT < 0) || ((objs.objs[i].getOVT() < conPVT) && 
			     (objs.objs[i].getOVT() >= 0)))
	  conPVT = objs.objs[i].getOVT();
      }
      CmiAssert((optPVT >= estGVT) || (optPVT == -1));
      CmiAssert((conPVT >= estGVT) || (conPVT == -1));
    }

  // pack PVT data
  umsg = new UpdateMsg;
  umsg->earlyTS = -1;
  //umsg->earlySends = umsg->earlyRecvs = 0;
  SendsAndRecvs->FindEarliestDiff(LastSendsAndRecvs, &(umsg->earlyTS));
  //  SendsAndRecvs->FindEarliest(&(umsg->earlyTS), &(umsg->earlySends), 
  //			      &(umsg->earlyRecvs)); 
  CmiAssert((umsg->earlyTS >= estGVT) || (umsg->earlyTS == -1));
  umsg->optPVT = optPVT;
  umsg->conPVT = conPVT;

  // send data to GVT estimation
  if (simdone) // transmit final info to GVT on PE 0
    g[0].computeGVT(umsg);              
  else {
    g[gvtTurn].computeGVT(umsg);           // transmit info to GVT
    gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
  }
  waitingForGVT = 1;
  SendsAndRecvs->CopyTable(LastSendsAndRecvs);
  objs.SetIdle(); // Set objects to idle
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

/// ENTRY: receive GVT estimate; wake up objects
void PVT::setGVT(GVTMsg *m)
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  estGVT = m->estGVT;
  simdone = m->done;
  CkFreeMsg(m);
  SendsAndRecvs->PurgeBelow(estGVT);
  if (!simdone) SendsAndRecvs->FileResiduals();
  objs.Commit();
  waitingForGVT = 0;
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

/// Register poser with PVT
int PVT::objRegister(int arrIdx, int safeTime, int sync, sim *myPtr)
{
  int i = objs.Insert(arrIdx, safeTime, sync, myPtr); // add to object list
  return(i*1000 + CkMyPe());                          // return unique PVT idx
}

// Unregister poser from PVT
void PVT::objRemove(int pvtIdx)
{
  int idx = (pvtIdx-CkMyPe())/1000;  // calculate local index from unique index
  objs.Delete(idx);                  // delete the object
}

/// Update send/recv table at timestamp
void PVT::objUpdate(int timestamp, int sr)
{
  CmiAssert(timestamp >= estGVT);
  CmiAssert((sr == SEND) || (sr == RECV));
  SendsAndRecvs->Insert(timestamp, sr);
}

/// Update PVT with safeTime and send/recv table at timestamp
void PVT::objUpdate(int pvtIdx, int safeTime, int timestamp, int sr)
{
  int index = (pvtIdx-CkMyPe())/1000;
  CmiAssert((timestamp >= estGVT) || (timestamp == -1));
  CmiAssert((safeTime >= estGVT) || (safeTime == -1));
  // minimize the non-idle OVT
  if ((safeTime >= 0) && 
      ((objs.objs[index].getOVT() > safeTime) || 
       (objs.objs[index].getOVT() < 0)))
    objs.objs[index].setOVT(safeTime);
  if ((sr == SEND) || (sr == RECV)) SendsAndRecvs->Insert(timestamp, sr);
  // sr could be -1 in which case we just ignore it here
}

/// Basic Constructor
GVT::GVT() 
{
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
#ifdef LB_ON
  nextLBstart = LB_SKIP - 1;
#endif
  estGVT = lastEarliest = inactiveTime = -1;
  lastSends = lastRecvs = inactive = 0;
  if (CkMyPe() == 0) { // start the PVT phase of the GVT algorithm
    CProxy_PVT p(ThePVT);
    p.startPhase(); // broadcast PVT calculation to all PVT branches
  }
}

// Used for Ccd calls; currently commented out
//void GVT::_runGVT(UpdateMsg *m) 
//{ 
//  CProxy_GVT g(TheGVT);
//  g[(CkMyPe() + 1)%CkNumPes()].runGVT(m);
//}

/// ENTRY: Run the GVT
void GVT::runGVT(UpdateMsg *m) 
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  estGVT = m->optPVT;
  lastEarliest = m->earlyTS;
  lastSends = m->earlySends;
  lastRecvs = m->earlyRecvs;
  inactive = m->inactive;
  inactiveTime = m->inactiveTime;
  nextLBstart = m->nextLB;
  CkFreeMsg(m);
  CProxy_PVT p(ThePVT);
  p.startPhase();  // start the next PVT phase of the GVT algorithm
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

/// ENTRY: Gathers PVT reports; calculates and broadcasts GVT to PVTs
void GVT::computeGVT(UpdateMsg *m)
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  CProxy_PVT p(ThePVT);
  CProxy_GVT g(TheGVT);
  GVTMsg *gmsg = new GVTMsg;
  int lastGVT = 0;
  static int optGVT = -1, conGVT = -1, done=0;
  static int earliestMsg=-1, earlySends=0, earlyRecvs=0;

  CmiAssert((m->earlyTS >= estGVT) || (m->earlyTS < 0));
  CmiAssert((m->optPVT >= estGVT) || (m->optPVT < 0));
  CmiAssert((m->conPVT >= estGVT) || (m->conPVT < 0));

  // see if message provides new min optGVT or conGVT
  if ((optGVT < 0) || ((m->optPVT >= 0) && (m->optPVT < optGVT)))
    optGVT = m->optPVT;
  if ((conGVT < 0) || ((m->conPVT >= 0) && (m->conPVT < conGVT)))
    conGVT = m->conPVT;
  // see if message provides new min earliestMsg
  //CkPrintf("m:earlyMsg=%d early#S=%d early#R=%d\n", m->earlyTS, m->earlySends, m->earlyRecvs);
  if (((m->earlyTS < earliestMsg) && (m->earlyTS >= 0)) 
      || (earliestMsg == -1)) {
    earliestMsg = m->earlyTS;
    //earlySends = m->earlySends;
    //earlyRecvs = m->earlyRecvs;
  }
  /*
  else if (m->earlyTS == earliestMsg) {
    earlySends += m->earlySends;
    earlyRecvs += m->earlyRecvs;
  }
  */
  CkFreeMsg(m);
  //CkPrintf("c:earlyMsg=%d early#S=%d early#R=%d\n", earliestMsg, earlySends, earlyRecvs);
  done++;

  if (done == CkNumPes()) { // all PVT reports are in
#ifdef POSE_STATS_ON
    localStats->GvtInc();
#endif
    done = 0;
    lastGVT = estGVT; // store previous estimate
    estGVT = -1;
    
    //CkPrintf("PRE: opt=%d con=%d lastGVT=%d early=%d #S=%d #R=%d last=%d #S=%d #R=%d et=%d\n", optGVT, conGVT, lastGVT, earliestMsg, earlySends, earlyRecvs, lastEarliest, lastSends, lastRecvs, POSE_endtime);

    // derive GVT estimate from min optimistic & conservative GVTs
    estGVT = optGVT;
    if ((conGVT >= 0) && (estGVT >= 0) && (conGVT < estGVT))  estGVT = conGVT;

    // Check if send/recv activity provides lower possible estimate
    /*
    if (earliestMsg >= 0) {
      if ((earliestMsg == lastEarliest) && (earlySends == lastSends) &&
	  (earlyRecvs == lastRecvs) && (earlySends == earlyRecvs)) {
	// no change to earliest S/R info from last GVT estimation
	lastEarliest = earliestMsg + 1; 
        lastSends = lastRecvs = 0;
      }
      else {
	lastEarliest = earliestMsg; 
        lastSends = earlySends; lastRecvs = earlyRecvs;
      }
    }
    else {
      lastEarliest = earliestMsg; 
      lastSends = earlySends; lastRecvs = earlyRecvs;
    }
    if (((lastEarliest < estGVT) && (lastEarliest >= 0)) || (estGVT < 0))
      estGVT = lastEarliest;
    */
    
    if ((earliestMsg >= 0) && ((earliestMsg < estGVT) || (estGVT < 0)))
      estGVT = earliestMsg;

    //CkPrintf("POST: opt=%d con=%d lastGVT=%d early=%d #S=%d #R=%d last=%d #S=%d #R=%d et=%d\n", optGVT, conGVT, lastGVT, earliestMsg, earlySends, earlyRecvs, lastEarliest, lastSends, lastRecvs, POSE_endtime);

    // check for inactivity
    if (((estGVT == lastGVT) || (estGVT < 0)) && (earliestMsg == -1)) {
      inactive++; 
      estGVT = lastGVT;
      if (inactive == 1) inactiveTime = lastGVT;
    }
    else if (estGVT < 0) {
      estGVT = lastGVT;
      inactive = 0;
    }
    else inactive = 0;

    // check the estimate
    CmiAssert(estGVT >= lastGVT); 
    
    //CkPrintf("[%d] New GVT = %d\n", CkMyPe(), estGVT);

    // check for termination conditions
    int term = 0;
    if ((estGVT >= POSE_endtime) && (POSE_endtime >= 0)) {
      CkPrintf("At endtime: %d\n", POSE_endtime);
      term = 1;
    }
    else if (inactive > 5) {
      CkPrintf("Simulation inactive at time: %d\n", inactiveTime);
      term = 1;
    }

    // report the new GVT estimate to all PVT branches
    gmsg->estGVT = estGVT;
    gmsg->done = term;
    if (term) {
      if (POSE_endtime >= 0) gmsg->estGVT = POSE_endtime + 1;
      else gmsg->estGVT++;
      CkPrintf("Final GVT = %d\n", gmsg->estGVT);
      p.setGVT(gmsg);
      POSE_stop();
    }
    else {
      p.setGVT(gmsg);

#ifdef LB_ON
      // perform load balancing
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(LB_TIMER);
#endif
      static int lb_skip = LB_SKIP;
      if (CkNumPes() > 1) {
	nextLBstart++;
	if (lb_skip == nextLBstart) {
	  TheLBG.calculateLocalLoad();
	  nextLBstart = 0;
	}
      }
#ifdef POSE_STATS_ON
      localStats->SwitchTimer(GVT_TIMER);
#endif
#endif

      // transmit data to start next GVT estimation on next GVT branch
      UpdateMsg *umsg = new UpdateMsg;
      umsg->optPVT = estGVT;
      //      umsg->earlyTS = lastEarliest;
      //      umsg->earlySends = lastSends;
      //      umsg->earlyRecvs = lastRecvs;
      umsg->inactive = inactive;
      umsg->inactiveTime = inactiveTime;
      umsg->nextLB = nextLBstart;
      g[(CkMyPe()+1) % CkNumPes()].runGVT(umsg);
    }

    // reset static data
    optGVT = conGVT = earliestMsg = -1;
    //earlySends = earlyRecvs = 0;
  }
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}
