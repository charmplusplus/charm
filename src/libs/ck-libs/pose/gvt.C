// File: gvt.C
// Implements the Global Virtual Time (GVT) algorithm; provides classes PVT
// and GVT.  PVT is a chare group (one branch per PE).  Objects interact with
// the local PVT branch.  PVT branches summarize object info and report to 
// the single GVT object, which broadcasts results to all PVT branches. 
// Last Modified: 06.24.02 by Terry L. Wilmarth

#include "pose.h"
#include "srtable.h"
#include "gvt.def.h"
#include "qd.h"

CkGroupID ThePVT;
CkGroupID TheGVT;

// Basic initializations
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
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

// Start a PVT/GVT cycle
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

  //CkPrintf(" ... PVT[%d] responding at startphase...\n", CkMyPe());
  if (waitingForGVT) {
    p[CkMyPe()].startPhase();  // start this later
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
    return;
  }

  // Step 1: Reset all flags and waken objects
  objs.SetIdle();            // set all objects to idle
  objs.Wake();               // wake all objects to get OVTs

  // Step 2: compute PVT
  optPVT = conPVT = -1;
  for (i=0; i<objs.numSpaces; i++)
    if (objs.objs[i].present) {
      if ((objs.objs[i].sync == OPTIMISTIC) && 
	  ((optPVT < 0) || 
	   ((objs.objs[i].ovt < optPVT) && 
	    (objs.objs[i].ovt >= 0))))  // check optPVT 
	optPVT = objs.objs[i].ovt;
      else if ((objs.objs[i].sync == CONSERVATIVE) &&
	       ((conPVT < 0) || 
	       ((objs.objs[i].ovt < conPVT) 
		&& (objs.objs[i].ovt >= 0))))  // check conPVT
	conPVT = objs.objs[i].ovt;
      if (!((optPVT >= estGVT) || (optPVT == -1)))
	CkPrintf("optPVT=%d estGVT=%d\n", optPVT, estGVT);
      CmiAssert((optPVT >= estGVT) || (optPVT == -1));
    }

  // Step 3: pack up PVT data to send to GVT
  umsg = SendsAndRecvs->packTable();
  CmiAssert((umsg->earlyTS >= estGVT) || (umsg->earlyTS == -1));
  CmiAssert((umsg->nextTS >= estGVT) || (umsg->nextTS == -1));
  umsg->optPVT = optPVT;
  umsg->conPVT = conPVT;
  if (simdone) {
    g[0].computeGVT(umsg);              // transmit final info to GVT
  }
  else {
    g[gvtTurn].computeGVT(umsg);           // transmit info to GVT
    gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
  }
  waitingForGVT = 1;
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

// Set estGVT on local branch and commit events (called by GVT)
void PVT::setGVT(GVTMsg *m)
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  simdone = m->done;
  estGVT = m->estGVT;
  CkFreeMsg(m);
  SendsAndRecvs->PurgeBelow(estGVT);
  if (!simdone)  SendsAndRecvs->FileResiduals();
  objs.Commit();
  waitingForGVT = 0;
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

// Register an object with the local PVT
int PVT::objRegister(int arrIdx, int safeTime, int sync, sim *myPtr)
{
  int i = objs.Insert(arrIdx, safeTime, sync, myPtr); // add to object list
  return(i*1000 + CkMyPe());                          // return unique PVT idx
}

// Unregister object from PVT
void PVT::objRemove(int pvtIdx)
{
  int idx = (pvtIdx-CkMyPe())/1000;  // calculate local index from unique index
  objs.Delete(idx);                  // delete the object
}

// Update sends/recvs arrays and residual bin with new message info
void PVT::objUpdate(int timestamp, int sr)
{
  CmiAssert(timestamp >= estGVT);
  if ((sr == SEND) || (sr == RECV)) {
    SendsAndRecvs->Insert(timestamp, sr);
    //    CkPrintf("Received %s at timestamp %d!\n", (sr==SEND)?"SEND":"RECV", 
    //	     timestamp);
  }
  else 
    CkPrintf("ERROR: PVT::objUpdate(%d, %d) has invalid sr value\n",
	     timestamp, sr);
}

// Update sends/recvs arrays with information about a send/recv at timestamp
// and/or update object's ovt with PVT during cycle
void PVT::objUpdate(int pvtIdx, int safeTime, int timestamp, int sr)
{
  int index = (pvtIdx-CkMyPe())/1000;

  CmiAssert((timestamp >= estGVT) || (timestamp == -1));
  // minimize the non-idle OVT
  CmiAssert((safeTime >= estGVT) || (safeTime == -1));
  if ((safeTime >= 0) && 
      ((objs.objs[index].ovt > safeTime) || (objs.objs[index].ovt < 0)))
    objs.objs[index].ovt = safeTime;

  if ((sr == SEND) || (sr == RECV)) {
    SendsAndRecvs->Insert(timestamp, sr);
    //    CkPrintf("Received %s at timestamp %d!\n", (sr==SEND)?"SEND":"RECV", 
    //	     timestamp);
  }
  // sr could be -1 in which case we just ignore it here
}

// Basic initializations
GVT::GVT() 
{
#ifdef POSE_STATS_ON
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  estGVT = -1;
  lastEarliest = lastSends = lastRecvs = inactive = 0;
  lastNextEarliest = lastNextSends = lastNextRecvs = 0;
#ifdef LB_ON
  nextLBstart = LB_SKIP - 1;
#endif
  if (CkMyPe() == 0)
    runGVT();  // get the GVT started on PE 0
}

// Used for Ccd calls; currently commented out
void GVT::_runGVT(UpdateMsg *m) 
{ 
  CProxy_GVT g(TheGVT);
  g[(CkMyPe() + 1)%CkNumPes()].runGVT(m);
}

void GVT::runGVT() 
{
  CProxy_PVT p(ThePVT);

  p.startPhase();  // start the PVT phase of the GVT algorithm
}

void GVT::runGVT(UpdateMsg *m) 
{
#ifdef POSE_STATS_ON
  localStats->TimerStart(GVT_TIMER);
#endif
  estGVT = m->optPVT;
  lastEarliest = m->earlyTS;
  lastSends = m->earlySends;
  lastRecvs = m->earlyRecvs;
  lastNextEarliest = m->nextTS;
  lastNextSends = m->nextSends;
  lastNextRecvs = m->nextRecvs;
  //nextLBstart = m->nextLB;
  CkFreeMsg(m);
  CProxy_PVT p(ThePVT);
  p.startPhase();  // start the PVT phase of the GVT algorithm
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

// Gathers PVT reports; computes GVT estimate and broadcasts to PVT branches
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
  static int earliestMsg, earlySends, earlyRecvs;
  static int nextEarliest, nextSends, nextRecvs;

  // process message
  if ((optGVT < 0) || ((m->optPVT >= 0) && (m->optPVT < optGVT)))
    optGVT = m->optPVT;
  if ((conGVT < 0) || ((m->conPVT >= 0) && (m->conPVT < conGVT)))
    conGVT = m->conPVT;
  if (earliestMsg == -1) { // first set of data received
    earliestMsg = m->earlyTS;
    earlySends = m->earlySends;
    earlyRecvs = m->earlyRecvs;
    nextEarliest = m->nextTS;
    nextSends = m->nextSends;
    nextRecvs = m->nextRecvs;
  }
  else { // this is not the first set of data
    // look at earliest first
    if ((m->earlyTS < earliestMsg) || (earliestMsg == -1)) {
      if ((earliestMsg < nextEarliest) || (nextEarliest == -1)) {
	nextEarliest = earliestMsg;
	nextSends = earlySends;
	nextRecvs = earlyRecvs;
      }
      else if (earliestMsg == nextEarliest) {
	nextSends += earlySends;
	nextRecvs += earlyRecvs;
      }
      earliestMsg = m->earlyTS;
      earlySends = m->earlySends;
      earlyRecvs = m->earlyRecvs;
    }
    else if (m->earlyTS == earliestMsg) {
      earlySends += m->earlySends;
      earlyRecvs += m->earlyRecvs;
    }
    else if ((m->earlyTS > earliestMsg) && (m->earlyTS < nextEarliest)) {
      nextEarliest = m->earlyTS;
      nextSends = m->earlySends;
      nextRecvs = m->earlyRecvs;
    }
    else if ((m->earlyTS > earliestMsg) && (m->earlyTS == nextEarliest)) {
      nextSends += m->earlySends;
      nextRecvs += m->earlyRecvs;
    }
    // now look at second earliest
    if ((m->nextTS < nextEarliest) || (nextEarliest == -1)) {
      nextEarliest = m->nextTS;
      nextSends = m->nextSends;
      nextRecvs = m->nextRecvs;
    }
    else if (m->nextTS == nextEarliest) {
      nextSends += m->nextSends;
      nextRecvs += m->nextRecvs;
    }
  }
  CkFreeMsg(m);
  done++;

  if (done == CkNumPes()) { // all PVT reports are in
#ifdef POSE_STATS_ON
    localStats->GvtInc();
#endif
    done = 0;
    lastGVT = estGVT; // store previous estimate
    estGVT = -1;
    
    // STEP 1: Derive estimate from optimistic & conservative GVTs; estimate
    //         is minimum of the two (discounting -1)
    if ((conGVT < 0) && (optGVT >= 0))  estGVT = optGVT;
    else if ((optGVT < 0) && (conGVT >= 0))  estGVT = conGVT;
    else if ((optGVT >= 0) && (conGVT >= 0)) {
	if (optGVT > conGVT) estGVT = optGVT;
	else estGVT = conGVT;
    }
    
    if (estGVT < 0) {
      inactive++; 
      if (inactive == 1) inactiveTime = lastGVT;
    }
    else inactive = 0;

    // STEP 2: Check if send/recv activity provides lower possible estimate
    if (earliestMsg < 0) earliestMsg = estGVT;
    else {
      if ((earliestMsg == lastEarliest) && (earlySends == lastSends) &&
	  (earlyRecvs == lastRecvs) && (earlySends == earlyRecvs)) {
	// no change to earliest S/R info from last GVT estimation
	earliestMsg = nextEarliest;
	earlySends = nextSends;
	earlyRecvs = nextRecvs;
      }
      if ((nextEarliest == lastNextEarliest) && (nextSends == lastNextSends) &&
	  (nextRecvs == lastNextRecvs) && (nextSends == nextRecvs)) {
	// no change to nextEarliest S/R info from last GVT estimation
	earliestMsg = nextEarliest++;
	earlySends = earlyRecvs = 0;
      }
      if (earliestMsg == nextEarliest) {
	nextEarliest = nextEarliest+1; nextSends = nextSends = 0;
      }
      else if (earliestMsg == nextEarliest + 1) {
	nextEarliest = nextEarliest+2; nextSends = nextSends = 0;
      }
      if ((earliestMsg > -1) && ((earliestMsg < estGVT) || (estGVT < 0)))
	estGVT = earliestMsg;
    }

    //CkPrintf("opt=%d con=%d lastGVT=%d earlyMsg=%d early#S=%d early#R=%d lastMsg=%d last#S=%d last#R=%d\n", optGVT, conGVT, lastGVT, earliestMsg, earlySends, earlyRecvs, lastEarliest, lastSends, lastRecvs);
    
    // STEP 3: In times of inactivity, GVT must be set to lastGVT
    if ((estGVT < 0) && (lastGVT < 0)) estGVT = 0;
    else if (estGVT < 0) estGVT = lastGVT;
    
    // STEP 4: If all has gone well, estimate >= previous estimate
    if ((estGVT < lastGVT) && (estGVT >= 0)) {
      CkPrintf("ERROR: new GVT estimate %d less than last one %d!\n",
	       estGVT, lastGVT);
      CkAbort("FATAL ERROR: GVT exiting...\n");
    }
    
    //CkPrintf("[%d] New GVT = %d\n", CkMyPe(), estGVT);

    // STEP 5: Check for termination conditions
    int term = 0;
    if ((estGVT >= POSE_endtime) && (POSE_endtime >= 0)) {
      CkPrintf("At endtime: %d\n", POSE_endtime);
      term = 1;
    }
    else if (inactive > 5) {
      CkPrintf("Simulation inactive at time: %d\n", inactiveTime);
      term = 1;
    }

    // STEP 6: Report the new GVT estimate to all PVT branches
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
      UpdateMsg *umsg = new UpdateMsg;
      umsg->optPVT = estGVT;
      umsg->earlyTS = earliestMsg;
      umsg->earlySends = earlySends;
      umsg->earlyRecvs = earlyRecvs;
      umsg->nextTS = nextEarliest;
      umsg->nextSends = nextSends;
      umsg->nextRecvs = nextRecvs;
      g[(CkMyPe()+1) % CkNumPes()].runGVT(umsg);
    }
    optGVT = conGVT = -1;
    earliestMsg = nextEarliest = -1;
  }
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}
