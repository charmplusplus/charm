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
  simdone = 0;
  SendsAndRecvs = new SRtable();
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
  optPVT = conPVT = -1;
  for (i=0; i<objs.getNumSpaces(); i++)
    if (objs.objs[i].isPresent()) {
      if (objs.objs[i].isOptimistic()) { // check optPVT 
	if ((optPVT < 0) || ((objs.objs[i].getOVT() < optPVT) && 
			     (objs.objs[i].getOVT() > -1))) {
	  optPVT = objs.objs[i].getOVT();
	  if ((objs.objs[i].getOVT() < estGVT) && (objs.objs[i].getOVT() > -1))
	    CkPrintf("Object %d has strange value %d < estGVT %d", i, 
		     objs.objs[i].getOVT(), estGVT);
	}
      }
      else if (objs.objs[i].isConservative()) { // check conPVT
	if ((conPVT < 0) || ((objs.objs[i].getOVT() < conPVT) && 
			     (objs.objs[i].getOVT() > -1)))
	  conPVT = objs.objs[i].getOVT();
      }
      if ((optPVT < estGVT) && (optPVT != -1) && (estGVT != -1))
	CkPrintf("optPVT=%d estGVT=%d\n", optPVT, estGVT);
      CmiAssert((optPVT >= estGVT) || (optPVT == -1) || (estGVT == -1));
      CmiAssert((conPVT >= estGVT) || (conPVT == -1) || (estGVT == -1));
    }

  // pack PVT data
  // (1) Find out the local PVT from optPVT and conPVT
  int pvt = optPVT;
  if ((conPVT < pvt) && (conPVT > -1)) pvt = conPVT;
  // (2) Find out how many timestamp send/recv records we need to send
  int counter = 0;
  SRentry *tmp = SendsAndRecvs->srs;
  while (tmp && ((tmp->timestamp() <= pvt) || (pvt == -1))) {
    counter++;
    tmp = tmp->next();
  }
  // (3) Create the message
  UpdateMsg *umsg = new(counter, 0) UpdateMsg;
  // (4) Pack the SRtable data into the message
  int index=0;
  tmp = SendsAndRecvs->srs;
  while (tmp && ((tmp->timestamp() <= pvt) || (pvt == -1))) {
    umsg->SRs[index] = *tmp;
    index++;
    tmp = tmp->next();
  }
  CmiAssert(index == counter);
  umsg->countSRs = counter;
  // (5) Pack the PVT info into the message
  umsg->optPVT = optPVT;
  umsg->conPVT = conPVT;
  umsg->runGVTflag = 0;

  // send data to GVT estimation
  if (simdone) // transmit final info to GVT on PE 0
    g[0].computeGVT(umsg);              
  else {
    g[gvtTurn].computeGVT(umsg);           // transmit info to GVT
    gvtTurn = (gvtTurn + 1) % CkNumPes();  // calculate next GVT location
  }
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
  static int nextQuanta = DOP_QUANTA;
#endif
  CProxy_PVT p(ThePVT);
  estGVT = m->estGVT;
  simdone = m->done;
  CkFreeMsg(m);
  SendsAndRecvs->PurgeBelow(estGVT);
  objs.Commit();
  p[CkMyPe()].startPhase();
#ifdef POSE_STATS_ON
  if (m->estGVT > nextQuanta) {
    nextQuanta += DOP_QUANTA;
    DOPcalc();
  }
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
  CmiAssert((timestamp >= estGVT) || (estGVT == -1));
  CmiAssert((sr == SEND) || (sr == RECV));
  SendsAndRecvs->Insert(timestamp, sr);
}

/// Update PVT with safeTime and send/recv table at timestamp
void PVT::objUpdate(int pvtIdx, int safeTime, int timestamp, int sr)
{
  int index = (pvtIdx-CkMyPe())/1000;
  //CmiAssert((timestamp >= estGVT) || (timestamp == -1) || (estGVT == -1));
  CmiAssert((safeTime >= estGVT) || (safeTime == -1));
  // minimize the non-idle OVT
  if ((safeTime > -1) && 
      ((objs.objs[index].getOVT() > safeTime) || 
       (objs.objs[index].getOVT() < 0)))
    objs.objs[index].setOVT(safeTime);
  //if ((sr == SEND) || (sr == RECV)) SendsAndRecvs->Insert(timestamp, sr);
  // sr could be -1 in which case we just ignore it here
}

void PVT::addToObjQdo(int pvtIdx, double t)
{
  int index = (pvtIdx-CkMyPe())/1000;
  objs.objs[index].addQdoTime(t);
}

void PVT::DOPcalc()
{
  double totalDoP=0.0, avg;
  static double sum=0.0;
  static int qcount=0;
  for (int i=0; i<objs.getNumObjs(); i++) {
    totalDoP += objs.objs[i].getQdo();
    objs.objs[i].resetQdo();
  }
  qcount++;
  sum += totalDoP;
  avg = sum/(double)qcount;
  CkPrintf("[%d] @ quanta %d worked %fs... AVG=%f\n", CkMyPe(), 
	   estGVT, totalDoP, avg);
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
  static int optGVT = -1, conGVT = -1, done=0;
  static int earliestMsg=-1;
  static SRentry *SRs = NULL;
  static int startOffset = 0;

  if (CkMyPe() != 0) startOffset = 1;
  if (m->runGVTflag == 1) done++;
  else {
    // see if message provides new min optGVT or conGVT
    if ((optGVT < 0) || ((m->optPVT > -1) && (m->optPVT < optGVT)))
      optGVT = m->optPVT;
    if ((conGVT < 0) || ((m->conPVT > -1) && (m->conPVT < conGVT)))
      conGVT = m->conPVT;
    // add send/recv info to SRs
    for (i=0; i<m->countSRs; i++) addSR(&SRs, m->SRs[i]);
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
    estGVT = -1;
    
    // derive GVT estimate from min optimistic & conservative GVTs
    estGVT = optGVT;
    if ((conGVT > -1) && (estGVT > -1) && (conGVT < estGVT))  estGVT = conGVT;

    // Check if send/recv activity provides lower possible estimate
    SRentry *tmp = SRs;
    int lastSR = -1;
    while (tmp && ((tmp->timestamp() < estGVT) || (estGVT == -1))) {
      lastSR = tmp->timestamp();
      if (tmp->sends() != tmp->recvs()) {
	earliestMsg = tmp->timestamp();
	break;
      }
      tmp = tmp->next();
    }
    if ((earliestMsg < estGVT) && (earliestMsg != -1)) estGVT = earliestMsg;
    else if ((earliestMsg == -1) && (lastSR != -1) && (estGVT == -1)
	     && (lastSR > lastGVT)) 
      estGVT = lastSR;

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
    //CkPrintf("opt=%d con=%d lastGVT=%d early=%d lastSR=%d et=%d\n", 
    //optGVT, conGVT, lastGVT, earliestMsg, lastSR, POSE_endtime);
    CmiAssert(estGVT >= lastGVT); 
    //if (estGVT % 100 == 0)
    //CkPrintf("[%d] New GVT = %d\n", CkMyPe(), estGVT);

    // check for termination conditions
    int term = 0;
    if ((estGVT >= POSE_endtime) && (POSE_endtime > -1)) {
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
      if (POSE_endtime > -1) gmsg->estGVT = POSE_endtime + 1;
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
    optGVT = conGVT = earliestMsg = -1;
    SRentry *cur = SRs;
    SRs = NULL;
    while (cur) {
      tmp = cur->next();
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
    (*SRs) = new SRentry(e.timestamp(), NULL);
    (*SRs)->setSends(e.sends());
    (*SRs)->setRecvs(e.recvs());
  }
  else {
    if (e.timestamp() < (*SRs)->timestamp()) { // goes before first entry
      (*SRs) = new SRentry(e.timestamp(), (*SRs));
      (*SRs)->setSends(e.sends());
      (*SRs)->setRecvs(e.recvs());
    }
    else if (e.timestamp() == (*SRs)->timestamp()) { // goes in first entry
      (*SRs)->setSends((*SRs)->sends() + e.sends());
      (*SRs)->setRecvs((*SRs)->recvs() + e.recvs());
    }
    else { // search for position
      tmp = (*SRs);
      while (tmp->next() && (e.timestamp() > tmp->next()->timestamp()))
	tmp = tmp->next();
      if (!tmp->next()) { // goes at end of SRs
	tmp->setNext(new SRentry(e.timestamp(), NULL));
	tmp->next()->setSends(tmp->next()->sends() + e.sends());
	tmp->next()->setRecvs(tmp->next()->recvs() + e.recvs());
      }
      else if (e.timestamp() == tmp->next()->timestamp()) { //goes in tmp->next
	tmp->next()->setSends(tmp->next()->sends() + e.sends());
	tmp->next()->setRecvs(tmp->next()->recvs() + e.recvs());
      }
      else { // goes after tmp but before tmp->next
	tmp->setNext(new SRentry(e.timestamp(), tmp->next()));
	tmp->next()->setSends(tmp->next()->sends() + e.sends());
	tmp->next()->setRecvs(tmp->next()->recvs() + e.recvs());
      }
    }
  }

}
