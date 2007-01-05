// Global Virtual Time estimation for POSE
#include "pose.h"
#include "srtable.h"
#include "gvt.def.h"
#include "qd.h"

CkGroupID ThePVT;
CkGroupID TheGVT;

/// Basic Constructor
PVT::PVT() 
{
  LBTurnInstrumentOff();
#ifdef POSE_COMM_ON
  //comm_debug = 1;
#endif
#ifndef CMK_OPTIMIZE
  localStats = (localStat *)CkLocalBranch(theLocalStats);
  if(pose_config.stats)
    {
      localStats->TimerStart(GVT_TIMER);
    }
#endif
  optPVT = conPVT = estGVT = POSE_UnsetTS;
  startPhaseActive = gvtTurn = simdone = 0;
  SendsAndRecvs = new SRtable();
  SendsAndRecvs->Initialize();
  specEventCount = eventCount = waitForFirst = 0;
  iterMin = POSE_UnsetTS;
  int P=CkNumPes(), N=CkMyPe();
  reportReduceTo =  -1;
  if ((N < P-2) && (N%2 == 1)) { //odd
    reportTo = N-1;
    reportsExpected = reportEnd = 0;
  }
  else if (N < P-2) { //even
    reportTo = N;
    reportsExpected = 2; 
    if (N == P-3)
      reportsExpected = 1;
    reportEnd = 0;
    if (N < (P-2)/2)
      reportReduceTo = P-2;
    else reportReduceTo = P-1;
  }
  if (N == P-2) {
    reportTo = N;
    reportEnd = 1;
    reportsExpected = 1 + (P-2)/4 + ((P-2)%4)/2;
  }
  else if (N == P-1) {
    reportTo = N;
    reportEnd = 1;
    if (P==1) reportsExpected = 1;
    else reportsExpected = 1 + (P-2)/4 + (P-2)%2;
  }
  //  CkPrintf("PE %d reports to %d, receives %d reports, reduces and sends to %d, and reports directly to GVT if %d = 1!\n", CkMyPe(), reportTo, reportsExpected, reportReduceTo, reportEnd);
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

void PVT::startPhaseExp(prioBcMsg *m) {
  startPhase(m);
}

/// ENTRY: runs the PVT calculation and reports to GVT
void PVT::startPhase(prioBcMsg *m) 
{
  CProxy_GVT g(TheGVT);
  CProxy_PVT p(ThePVT);
  register int i;

  if (startPhaseActive) return;
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(GVT_TIMER);
#endif
  startPhaseActive = 1;
  if (m->bc) {
    prioBcMsg *startMsg = new (8*sizeof(POSE_TimeType)) prioBcMsg;
    startMsg->bc = 0;
    *((POSE_TimeType *)CkPriorityPtr(startMsg)) = 1-POSE_TimeMax;
    CkSetQueueing(startMsg, CK_QUEUEING_IFIFO); 
    p.startPhaseExp(startMsg);
  }

  objs.Wake(); // wake objects to make sure all have reported
  // compute PVT
  optPVT = conPVT = POSE_UnsetTS;
  int end = objs.getNumSpaces();
  for (i=0; i<end; i++)
    if (objs.objs[i].isPresent()) {
      if (objs.objs[i].isOptimistic()) { // check optPVT 
	if ((optPVT < 0) || ((objs.objs[i].getOVT() < optPVT) && 
			     (objs.objs[i].getOVT() > POSE_UnsetTS))) {
	  optPVT = objs.objs[i].getOVT();
	  CkAssert(simdone || ((objs.objs[i].getOVT() >= estGVT) ||
			       (objs.objs[i].getOVT() == POSE_UnsetTS)));
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

  // (1) Find out the local PVT from optPVT and conPVT
  POSE_TimeType pvt = optPVT;
  if ((conPVT < pvt) && (conPVT > POSE_UnsetTS)) pvt = conPVT;
  if ((iterMin < pvt) && (iterMin > POSE_UnsetTS)) pvt = iterMin;
  if (waitForFirst) {
    waitForFirst = 0;
    if (pvt == POSE_UnsetTS)
      SendsAndRecvs->Restructure(estGVT, estGVT, POSE_UnsetTS);
    else
      SendsAndRecvs->Restructure(estGVT, pvt, POSE_UnsetTS);
  }

  //  CkPrintf("[%d] pvt=%d gvt=%d optPVT=%d iterMin=%d\n", CkMyPe(), pvt, estGVT, optPVT, iterMin);
  POSE_TimeType xt;
  if (pvt == POSE_UnsetTS) { // all are idle; find max ovt
    POSE_TimeType maxOVT = POSE_UnsetTS;
    for (i=0; i<end; i++)
      if (objs.objs[i].isPresent()) {
	xt = objs.objs[i].getOVT2();
	if (xt > maxOVT)
	  maxOVT = xt;
      }
    if (maxOVT > estGVT)
      pvt = maxOVT;
  }
  
  // (2) Pack the SRtable data into the message
  POSE_TimeType maxSR;
  UpdateMsg *um = SendsAndRecvs->PackTable(pvt, &maxSR);
  // (3) Add the PVT info to the message
  um->optPVT = pvt;
  um->conPVT = conPVT;
  um->maxSR = maxSR;
  um->runGVTflag = 0;

  if (um->numEntries > 0) {
    //CkPrintf("PE %d has %d SRs reported to GVT; earliest=%d pvt=%d\n", CkMyPe(), um->numEntries, um->SRs[0].timestamp, pvt);
  }
  // send data to GVT estimation
  p[reportTo].reportReduce(um);

  /*
  if (simdone) // transmit final info to GVT on PE 0
    g[0].computeGVT(um);              
  else {
    g[gvtTurn].computeGVT(um);           // transmit info to GVT
    gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
  }
  */
  objs.SetIdle(); // Set objects to idle
  iterMin = POSE_UnsetTS;
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

/// ENTRY: receive GVT estimate; wake up objects
void PVT::setGVT(GVTMsg *m)
{
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(GVT_TIMER);
#endif
  CProxy_PVT p(ThePVT);
  CkAssert(m->estGVT >= estGVT);
  estGVT = m->estGVT;
  int i, end = objs.getNumSpaces();
#ifdef POSE_COMM_ON  
  //PrioStreaming *pstrat = (PrioStreaming *)(POSE_commlib_insthndl.getStrategy());
  //pstrat->setBasePriority((estGVT+10) - POSE_TimeMax);
  //pstrat->setBasePriority(estGVT+10);
#endif
  simdone = m->done;
  CkFreeMsg(m);
  waitForFirst = 1;
  objs.Commit();
  startPhaseActive = 0;
  prioBcMsg *startMsg = new (8*sizeof(int)) prioBcMsg;
  startMsg->bc = 1;
  *((int *)CkPriorityPtr(startMsg)) = 0;
  CkSetQueueing(startMsg, CK_QUEUEING_IFIFO); 
  p[CkMyPe()].startPhase(startMsg);
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

/// Register poser with PVT
int PVT::objRegister(int arrIdx, POSE_TimeType safeTime, int sync, sim *myPtr)
{
  int i = objs.Insert(arrIdx, -1, sync, myPtr); // add to object list
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
#ifndef CMK_OPTIMIZE
  int tstat = localStats->TimerRunning();
  if(pose_config.stats){
    if (tstat)
      localStats->SwitchTimer(GVT_TIMER);
    else
      localStats->TimerStart(GVT_TIMER);
  }
#endif
  //if ((timestamp < estGVT) && (estGVT > POSE_UnsetTS))
  //CkPrintf("timestamp=%d estGVT=%d simdone=%d sr=%d\n", timestamp, estGVT, simdone, sr);
  CmiAssert(simdone || (timestamp >= estGVT) || (estGVT == POSE_UnsetTS));
  CmiAssert((sr == SEND) || (sr == RECV));
  if ((estGVT > POSE_UnsetTS) && 
      ((timestamp < iterMin) || (iterMin == POSE_UnsetTS))) 
    iterMin = timestamp;
  if (waitForFirst) {
    waitForFirst = 0;
    SendsAndRecvs->Restructure(estGVT, timestamp, sr);
  }
  else SendsAndRecvs->Insert(timestamp, sr);
#ifndef CMK_OPTIMIZE
  if(pose_config.stats){
    if (tstat)
      localStats->SwitchTimer(tstat);
    else
      localStats->TimerStop();
  }
#endif

}

/// Update PVT with safeTime
void PVT::objUpdateOVT(int pvtIdx, POSE_TimeType safeTime, POSE_TimeType ovt)
{
  int index = (pvtIdx-CkMyPe())/1000;
  // minimize the non-idle OVT
  //  if ((safeTime < estGVT) && (safeTime > POSE_UnsetTS)) 
  //CkPrintf("safeTime=%d estGVT=%d\n", safeTime, estGVT);
  CmiAssert(simdone || (safeTime >= estGVT) || (safeTime == POSE_UnsetTS));
  if ((safeTime == POSE_UnsetTS) && (objs.objs[index].getOVT2() < ovt))
    objs.objs[index].setOVT2(ovt);
  if ((safeTime > POSE_UnsetTS) && 
      ((objs.objs[index].getOVT() > safeTime) || 
       (objs.objs[index].getOVT() == POSE_UnsetTS)))
    objs.objs[index].setOVT(safeTime);
}

/// Reduction point for PVT reports
void PVT::reportReduce(UpdateMsg *m)
{
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(GVT_TIMER);
#endif
  CProxy_PVT p(ThePVT);
  CProxy_GVT g(TheGVT);
  POSE_TimeType lastGVT = 0, maxSR=0;
  static POSE_TimeType optGVT = POSE_UnsetTS, conGVT = POSE_UnsetTS;
  static int done=0;
  static SRentry *SRs = NULL;

  // see if message provides new min optGVT or conGVT
  if ((optGVT < 0) || ((m->optPVT > POSE_UnsetTS) && (m->optPVT < optGVT)))
    optGVT = m->optPVT;
  if (m->maxSR > maxSR)
    maxSR = m->maxSR;
  addSR(&SRs, m->SRs, optGVT, m->numEntries);
  done++;
  CkFreeMsg(m);

  if (done == reportsExpected) { // all PVT reports are in
    UpdateMsg *um;
    int entryCount = 0;
    // pack data into um
    SRentry *tmp = SRs;
    while (tmp && ((tmp->timestamp <= optGVT) || (optGVT == POSE_UnsetTS))
	   && (tmp->sends != tmp->recvs)) {
      entryCount++;
      tmp = tmp->next;
    }
    um = new (entryCount * sizeof(SRentry), 0) UpdateMsg;
    tmp = SRs;
    int i=0;
    while (tmp && ((tmp->timestamp <= optGVT) || (optGVT == POSE_UnsetTS))
	   && (tmp->sends != tmp->recvs)) {
      um->SRs[i] = *tmp;
      tmp = tmp->next;
      i++;
    }
    um->numEntries = entryCount;
    um->optPVT = optGVT;
    um->conPVT = conGVT;
    um->maxSR = maxSR;
    um->runGVTflag = 0;

    if (reportEnd) { //send to computeGVT
      if (simdone) // transmit final info to GVT on PE 0
	g[0].computeGVT(um);              
      else {
	g[gvtTurn].computeGVT(um);           // transmit info to GVT
	gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
      }
    }
    else { //send to pvt reportReduceTo
      p[reportReduceTo].reportReduce(um);
    }

    // reset static data
    optGVT = conGVT = POSE_UnsetTS;
    SRentry *cur = SRs;
    SRs = NULL;
    while (cur) {
      tmp = cur->next;
      delete cur;
      cur = tmp;
    }
    done = 0;
  }
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

/// Basic Constructor
GVT::GVT() 
{
#ifndef CMK_OPTIMIZE
  localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
#ifndef SEQUENTIAL_POSE
  if(pose_config.lb_on)
    nextLBstart = pose_config.lb_skip - 1;
#endif
  estGVT = lastEarliest = inactiveTime = POSE_UnsetTS;
  lastSends = lastRecvs = inactive = 0;
  reportsExpected = 1;
  if (CkNumPes() >= 2) reportsExpected = 2;
    
  //  CkPrintf("GVT expects %d reports!\n", reportsExpected);
  if (CkMyPe() == 0) { // start the PVT phase of the GVT algorithm
    CProxy_PVT p(ThePVT);
    prioBcMsg *startMsg = new (8*sizeof(int)) prioBcMsg;
    startMsg->bc = 1;
    *((int *)CkPriorityPtr(startMsg)) = 0;
    CkSetQueueing(startMsg, CK_QUEUEING_IFIFO); 
    p.startPhase(startMsg); // broadcast PVT calculation to all PVT branches
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
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(GVT_TIMER);
#endif
  estGVT = m->optPVT;
  inactive = m->inactive;
  inactiveTime = m->inactiveTime;
  nextLBstart = m->nextLB;
  CProxy_GVT g(TheGVT);
  m->runGVTflag = 1;
  g[CkMyPe()].computeGVT(m);  // start the next PVT phase of the GVT algorithm
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

/// ENTRY: Gathers PVT reports; calculates and broadcasts GVT to PVTs
void GVT::computeGVT(UpdateMsg *m)
{
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStart(GVT_TIMER);
#endif
  CProxy_PVT p(ThePVT);
  CProxy_GVT g(TheGVT);
  GVTMsg *gmsg = new GVTMsg;
  POSE_TimeType lastGVT = 0, earliestMsg = POSE_UnsetTS, 
    earlyAny = POSE_UnsetTS;
  static POSE_TimeType optGVT = POSE_UnsetTS, conGVT = POSE_UnsetTS;
  static int done=0;
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
    if (m->maxSR > earlyAny) 
      earlyAny = m->maxSR;
    // add send/recv info to SRs
    /*    if (m->numEntries > 0)
      CkPrintf("GVT recv'd %d SRs from a PE, earliest=%d\n", m->numEntries, 
      m->SRs[0].timestamp);*/
    addSR(&SRs, m->SRs, optGVT, m->numEntries);
    done++;
  }
  CkFreeMsg(m);

  if (done == reportsExpected+startOffset) { // all PVT reports are in
#ifndef CMK_OPTIMIZE
    if(pose_config.stats)
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
    /*    if (SRs) SRs->dump();
	  else CkPrintf("No SRs reported to GVT!\n");*/
    SRentry *tmp = SRs;
    POSE_TimeType lastSR = POSE_UnsetTS;
    while (tmp && ((tmp->timestamp <= estGVT) || (estGVT == POSE_UnsetTS))) {
      lastSR = tmp->timestamp;
      if (tmp->sends != tmp->recvs) {
	earliestMsg = tmp->timestamp;
	break;
      }
      tmp = tmp->next;
    }
    /*    if ((earliestMsg > POSE_UnsetTS) || (earlyAny > POSE_UnsetTS))
	  CkPrintf("GVT: earlyDiff=%d earlyAny=%d estGVT was %d.\n", earliestMsg, earlyAny, estGVT);*/
    if (((earliestMsg < estGVT) && (earliestMsg != POSE_UnsetTS)) ||
	(estGVT == POSE_UnsetTS))
      estGVT = earliestMsg;
    if ((lastSR != POSE_UnsetTS) && (estGVT == POSE_UnsetTS) && 
	(lastSR > lastGVT))
      estGVT = lastSR;

    // check for inactivity
    if ((optGVT == POSE_UnsetTS) && (earliestMsg == POSE_UnsetTS)) {
      inactive++;
      /*
      if (inactive == 1) {
	CkPrintf("[%d] Inactive... calling CkWaitQD...\n", CkMyPe());
	CkWaitQD();
	CkPrintf("[%d] Back from CkWaitQD...\n", CkMyPe());
      }
      */
      estGVT = lastGVT;
      if (inactive == 1) inactiveTime = lastGVT;
    }
    else if (estGVT < 0) {
      estGVT = lastGVT;
      inactive = 0;
    }
    else inactive = 0;

    // check the estimate
    //CkPrintf("opt=%d con=%d lastGVT=%d early=%d lastSR=%d et=%d\n", optGVT, conGVT, lastGVT, earliestMsg, lastSR, POSE_endtime);
    CmiAssert(estGVT >= lastGVT); 
    //if (estGVT % 1000 == 0)
    //CkPrintf("[%d] New GVT = %d\n", CkMyPe(), estGVT);
    //CkPrintf("[%d] New GVT = %lld\n", CkMyPe(), estGVT);

    // check for termination conditions
    int term = 0;
    if ((estGVT >= POSE_endtime) && (POSE_endtime > POSE_UnsetTS)) {
#if USE_LONG_TIMESTAMPS      
      CkPrintf("At endtime: %lld\n", POSE_endtime);
#else
      CkPrintf("At endtime: %d\n", POSE_endtime);
#endif
      term = 1;
    }
    else if (inactive > 2) {
#if USE_LONG_TIMESTAMPS      
      CkPrintf("Simulation inactive at time: %lld\n", inactiveTime);
#else
      CkPrintf("Simulation inactive at time: %d\n", inactiveTime);
#endif
      term = 1;
    }

    // report the last new GVT estimate to all PVT branches
    gmsg->estGVT = estGVT;
    gmsg->done = term;
    if (term) {
      //if (POSE_endtime > POSE_UnsetTS) gmsg->estGVT = POSE_endtime + 1;
      // else gmsg->estGVT++;
#if USE_LONG_TIMESTAMPS      
      CkPrintf("Final GVT = %lld\n", gmsg->estGVT);
#else
      CkPrintf("Final GVT = %d\n", gmsg->estGVT);
#endif
      p.setGVT(gmsg);
      POSE_stop();
    }
    else {
      p.setGVT(gmsg);

      if(pose_config.lb_on)
	{
	  // perform load balancing
#ifndef CMK_OPTIMIZE
	  if(pose_config.stats)
	    localStats->SwitchTimer(LB_TIMER);
#endif
	  static int lb_skip = pose_config.lb_skip;
	  if (CkNumPes() > 1) {
	    nextLBstart++;
	    if (lb_skip == nextLBstart) {
	      TheLBG.calculateLocalLoad();
	      nextLBstart = 0;
	    }
	  }
#ifndef CMK_OPTIMIZE
	  if(pose_config.stats)
	    localStats->SwitchTimer(GVT_TIMER);
#endif
	}

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
    optGVT = conGVT = POSE_UnsetTS;
    SRentry *cur = SRs;
    SRs = NULL;
    while (cur) {
      tmp = cur->next;
      delete cur;
      cur = tmp;
    }
  }
#ifndef CMK_OPTIMIZE
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

void GVT::addSR(SRentry **SRs, SRentry *e, POSE_TimeType og, int ne)
{
  register int i;
  SRentry *tab = (*SRs);
  SRentry *tmp = tab;

  for (i=0; i<ne; i++) {
    if ((e[i].timestamp < og) || (og == POSE_UnsetTS)) {
      if (!tmp) { // no entries yet
	tab = new SRentry(e[i].timestamp, (SRentry *)NULL);
	tab->sends = e[i].sends;
	tab->recvs = e[i].recvs;
	tmp = tab;
	*SRs = tmp;
      }
      else {
	if (e[i].timestamp < tmp->timestamp) { // goes before tmp
	  CkAssert(tmp == *SRs);
	  tab = new SRentry(e[i].timestamp, tmp);
	  tab->sends = e[i].sends;
	  tab->recvs = e[i].recvs;
	  tmp = tab;
	  *SRs = tmp;
	}
	else if (e[i].timestamp == tmp->timestamp) { // goes in first entr
	  tmp->sends = tmp->sends + e[i].sends;
	  tmp->recvs = tmp->recvs + e[i].recvs;
	}
	else { // search for position
	  while (tmp->next && (e[i].timestamp > tmp->next->timestamp))
	    tmp = tmp->next;
	  if (!tmp->next) { // goes at end of SRs
	    tmp->next = new SRentry(e[i].timestamp, (SRentry *)NULL);
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	  else if (e[i].timestamp == tmp->next->timestamp) {//goes in tmp->next
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	  else { // goes after tmp but before tmp->next
	    tmp->next = new SRentry(e[i].timestamp, tmp->next);
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	}
      }
    }
    else break;
  }
}

void PVT::addSR(SRentry **SRs, SRentry *e, POSE_TimeType og, int ne)
{
  register int i;
  SRentry *tab = (*SRs);
  SRentry *tmp = tab;

  for (i=0; i<ne; i++) {
    if ((e[i].timestamp < og) || (og == POSE_UnsetTS)) {
      if (!tmp) { // no entries yet
	tab = new SRentry(e[i].timestamp, (SRentry *)NULL);
	tab->sends = e[i].sends;
	tab->recvs = e[i].recvs;
	tmp = tab;
	*SRs = tmp;
      }
      else {
	if (e[i].timestamp < tmp->timestamp) { // goes before tmp
	  CkAssert(tmp == *SRs);
	  tab = new SRentry(e[i].timestamp, tmp);
	  tab->sends = e[i].sends;
	  tab->recvs = e[i].recvs;
	  tmp = tab;
	  *SRs = tmp;
	}
	else if (e[i].timestamp == tmp->timestamp) { // goes in first entr
	  tmp->sends = tmp->sends + e[i].sends;
	  tmp->recvs = tmp->recvs + e[i].recvs;
	}
	else { // search for position
	  while (tmp->next && (e[i].timestamp > tmp->next->timestamp))
	    tmp = tmp->next;
	  if (!tmp->next) { // goes at end of SRs
	    tmp->next = new SRentry(e[i].timestamp, (SRentry *)NULL);
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	  else if (e[i].timestamp == tmp->next->timestamp) {//goes in tmp->next
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	  else { // goes after tmp but before tmp->next
	    tmp->next = new SRentry(e[i].timestamp, tmp->next);
	    tmp->next->sends = tmp->next->sends + e[i].sends;
	    tmp->next->recvs = tmp->next->recvs + e[i].recvs;
	    tmp = tmp->next;
	  }
	}
      }
    }
    else break;
  }
}
