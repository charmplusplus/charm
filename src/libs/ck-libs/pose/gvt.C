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
  optPVT = conPVT = estGVT = POSE_UnsetTS;
  simdone = 0;
  SendsAndRecvs = new SRtable();
  SendsAndRecvs->Initialize();
  waitForFirst = 0;
  iterMin = POSE_UnsetTS;
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
  CProxy_GVT g(TheGVT);
  static int gvtTurn = 0;
  int i;

  objs.Wake(); // wake objects to make sure all have reported

  // compute PVT
  optPVT = conPVT = POSE_UnsetTS;
  for (i=0; i<objs.getNumSpaces(); i++)
    if (objs.objs[i].isPresent()) {
      if (objs.objs[i].isOptimistic()) { // check optPVT 
	if ((optPVT < 0) || ((objs.objs[i].getOVT() < optPVT) && 
			     (objs.objs[i].getOVT() > POSE_UnsetTS))) {
	  optPVT = objs.objs[i].getOVT();
	  CkAssert((objs.objs[i].getOVT() >= estGVT) ||
		   (objs.objs[i].getOVT() == POSE_UnsetTS));
	}
      }
      else if (objs.objs[i].isConservative()) { // check conPVT
	if ((conPVT < 0) || ((objs.objs[i].getOVT() < conPVT) && 
			     (objs.objs[i].getOVT() > POSE_UnsetTS)))
	  conPVT = objs.objs[i].getOVT();
      }
      CkAssert(simdone || (optPVT >= estGVT)||(optPVT == POSE_UnsetTS)||(estGVT == POSE_UnsetTS));
      CkAssert(simdone || (conPVT >= estGVT)||(conPVT == POSE_UnsetTS)||(estGVT == POSE_UnsetTS));
    }

  // pack PVT data
  // (1) Find out the local PVT from optPVT and conPVT
  int pvt = optPVT;
  if ((conPVT < pvt) && (conPVT > POSE_UnsetTS)) pvt = conPVT;
  if ((iterMin < pvt) && (iterMin > POSE_UnsetTS)) pvt = iterMin;
  if (waitForFirst) {
    waitForFirst = 0;
    SendsAndRecvs->Restructure(estGVT, pvt, POSE_UnsetTS);
  }
  // (2) Pack the SRtable data into the message
  UpdateMsg *um = SendsAndRecvs->PackTable(pvt);
  // (3) Add the PVT info to the message
  um->optPVT = pvt;
  um->conPVT = conPVT;
  um->runGVTflag = 0;

  // send data to GVT estimation
  if (simdone) // transmit final info to GVT on PE 0
    g[0].computeGVT(um);              
  else {
    g[gvtTurn].computeGVT(um);           // transmit info to GVT
    gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
  }
  objs.SetIdle(); // Set objects to idle
  iterMin = POSE_UnsetTS;
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
  CProxy_PVT p(ThePVT);
  estGVT = m->estGVT;
  simdone = m->done;
  CkFreeMsg(m);
  waitForFirst = 1;
  objs.Commit();
  p[CkMyPe()].startPhase();
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
void PVT::objUpdate(POSE_TimeType timestamp, int sr)
{
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (tstat)
    localStats->SwitchTimer(GVT_TIMER);
  else
    localStats->TimerStart(GVT_TIMER);
#endif
  CmiAssert(simdone || (timestamp >= estGVT) || (estGVT == POSE_UnsetTS));
  CmiAssert((sr == SEND) || (sr == RECV));
  if ((timestamp < iterMin) || (iterMin == POSE_UnsetTS)) iterMin = timestamp;
  if (waitForFirst) {
    waitForFirst = 0;
    SendsAndRecvs->Restructure(estGVT, timestamp, sr);
  }
  else { 
    SendsAndRecvs->Insert(timestamp, sr);
  }
  //CkPrintf("[%d] %s at %d\n", CkMyPe(), (sr == SEND)?"SEND":"RECV", timestamp);
#ifdef POSE_STATS_ON
  if (tstat)
    localStats->SwitchTimer(tstat);
  else
    localStats->TimerStop();
#endif

}

/// Update PVT with safeTime and send/recv table at timestamp
void PVT::objUpdate(int pvtIdx, POSE_TimeType safeTime, POSE_TimeType timestamp, int sr)
{
  int index = (pvtIdx-CkMyPe())/1000;

  //CmiAssert((timestamp >= estGVT) || (timestamp == POSE_UnsetTS) || (estGVT == POSE_UnsetTS));
  CmiAssert(simdone || (safeTime >= estGVT) || (safeTime == POSE_UnsetTS));
  // minimize the non-idle OVT
  if ((safeTime > POSE_UnsetTS) && 
      ((objs.objs[index].getOVT() > safeTime) || 
       (objs.objs[index].getOVT() < 0)))
    objs.objs[index].setOVT(safeTime);
  //CkPrintf("[%d] %d's safeTime is %d\n", CkMyPe(), index, safeTime);
  //if ((sr == SEND) || (sr == RECV)) SendsAndRecvs->Insert(timestamp, sr);
  // sr could be POSE_UnsetTS in which case we just ignore it here
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
  estGVT = lastEarliest = inactiveTime = POSE_UnsetTS;
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
  inactive = m->inactive;
  inactiveTime = m->inactiveTime;
  nextLBstart = m->nextLB;
  CProxy_GVT g(TheGVT);
  m->runGVTflag = 1;
  g[CkMyPe()].computeGVT(m);  // start the next PVT phase of the GVT algorithm
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
  int lastGVT = 0, i;
  static int optGVT = POSE_UnsetTS, conGVT = POSE_UnsetTS, done=0;
  static int earliestMsg=POSE_UnsetTS;
  static SRentry *SRs = NULL;
  static int startOffset = 0;

  if (CkMyPe() != 0) startOffset = 1;
  if (m->runGVTflag == 1) done++;
  else {
    // see if message provides new min optGVT or conGVT
    if ((optGVT < 0) || ((m->optPVT > POSE_UnsetTS) && (m->optPVT < optGVT)))
      optGVT = m->optPVT;
    if ((conGVT < 0) || ((m->conPVT > POSE_UnsetTS) && (m->conPVT < conGVT)))
      conGVT = m->conPVT;
    // add send/recv info to SRs
    for (i=0; i<m->numEntries; i++) {
      if ((m->SRs[i].timestamp < optGVT) || (optGVT == POSE_UnsetTS))
	addSR(&SRs, m->SRs[i]);
      else i = m->numEntries;
    }
    done++;
  }
  CkFreeMsg(m);

  if (done == CkNumPes()+startOffset) { // all PVT reports are in
#ifdef POSE_STATS_ON
    localStats->GvtInc();
#endif
    done = 0;
    startOffset = 1;
    lastGVT = estGVT; // store previous estimate
    if (lastGVT < 0) lastGVT = 0;
    estGVT = POSE_UnsetTS;
    
    // derive GVT estimate from min optimistic & conservative GVTs
    estGVT = optGVT;
    if ((conGVT > POSE_UnsetTS) && (estGVT > POSE_UnsetTS) && (conGVT < estGVT))  estGVT = conGVT;

    // Check if send/recv activity provides lower possible estimate
    SRentry *tmp = SRs;
    int lastSR = POSE_UnsetTS;
    while (tmp && ((tmp->timestamp < estGVT) || (estGVT == POSE_UnsetTS))) {
      lastSR = tmp->timestamp;
      if (tmp->sends != tmp->recvs) {
	earliestMsg = tmp->timestamp;
	break;
      }
      tmp = tmp->next;
    }
    if (((earliestMsg < estGVT) && (earliestMsg != POSE_UnsetTS)) ||
	(estGVT == POSE_UnsetTS))
      estGVT = earliestMsg;
    if ((lastSR != POSE_UnsetTS) && (estGVT == POSE_UnsetTS) && 
	(lastSR > lastGVT))
      estGVT = lastSR;

    // check for inactivity
    if ((optGVT == POSE_UnsetTS) && (earliestMsg == POSE_UnsetTS) && 
	(lastSR == POSE_UnsetTS)) {
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
    //CkPrintf("opt=%d con=%d lastGVT=%d early=%d lastSR=%d et=%d\n", 
    //optGVT, conGVT, lastGVT, earliestMsg, lastSR, POSE_endtime);
    CmiAssert(estGVT >= lastGVT); 
    //if (estGVT % 100 == 0)
    //CkPrintf("[%d] New GVT = %d\n", CkMyPe(), estGVT);

    // check for termination conditions
    int term = 0;
    if ((estGVT > POSE_endtime) && (POSE_endtime > POSE_UnsetTS)) {
      CkPrintf("At endtime: %d\n", POSE_endtime);
      term = 1;
    }
    else if (inactive > 5) {
      CkPrintf("Simulation inactive at time: %d\n", inactiveTime);
      term = 1;
    }

    // report the last new GVT estimate to all PVT branches
    gmsg->estGVT = estGVT;
    gmsg->done = term;
    if (term) {
      if (POSE_endtime > POSE_UnsetTS) gmsg->estGVT = POSE_endtime + 1;
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
      umsg->inactive = inactive;
      umsg->inactiveTime = inactiveTime;
      umsg->nextLB = nextLBstart;
      umsg->runGVTflag = 0;
      g[(CkMyPe()+1) % CkNumPes()].runGVT(umsg);
    }

    // reset static data
    optGVT = conGVT = earliestMsg = POSE_UnsetTS;
    SRentry *cur = SRs;
    SRs = NULL;
    while (cur) {
      tmp = cur->next;
      delete cur;
      cur = tmp;
    }
  }
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
}

void GVT::addSR(SRentry **SRs, SRentry e)
{
  SRentry *tmp;
  if (!(*SRs)) { // no entries yet
    (*SRs) = new SRentry(e.timestamp, (SRentry *)NULL);
    (*SRs)->sends = e.sends;
    (*SRs)->recvs = e.recvs;
  }
  else {
    if (e.timestamp < (*SRs)->timestamp) { // goes before first entry
      (*SRs) = new SRentry(e.timestamp, (*SRs));
      (*SRs)->sends = e.sends;
      (*SRs)->recvs = e.recvs;
    }
    else if (e.timestamp == (*SRs)->timestamp) { // goes in first entry
      (*SRs)->sends = (*SRs)->sends + e.sends;
      (*SRs)->recvs = (*SRs)->recvs + e.recvs;
    }
    else { // search for position
      tmp = (*SRs);
      while (tmp->next && (e.timestamp > tmp->next->timestamp))
	tmp = tmp->next;
      if (!tmp->next) { // goes at end of SRs
	tmp->next = new SRentry(e.timestamp, (SRentry *)NULL);
	tmp->next->sends = tmp->next->sends + e.sends;
	tmp->next->recvs = tmp->next->recvs + e.recvs;
      }
      else if (e.timestamp == tmp->next->timestamp) { //goes in tmp->next
	tmp->next->sends = tmp->next->sends + e.sends;
	tmp->next->recvs = tmp->next->recvs + e.recvs;
      }
      else { // goes after tmp but before tmp->next
	tmp->next = new SRentry(e.timestamp, tmp->next);
	tmp->next->sends = tmp->next->sends + e.sends;
	tmp->next->recvs = tmp->next->recvs + e.recvs;
      }
    }
  }

}
