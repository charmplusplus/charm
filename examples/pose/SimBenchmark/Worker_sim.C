#include "Worker_sim.h"
#include "Worker.def.h"

#include <math.h>
#include <math.h>
worker::worker(WorkerData *m){
#ifdef POSE_STATS_ON
  localStats->TimerStart(SIM_TIMER);
#endif
#ifdef LB_ON  
  LBgroup *localLBG = TheLBG.ckLocalBranch();

#endif  
  myStrat = new adapt3();
  m->parent = this;
  m->str = myStrat;
  POSE_TimeType _ts = m->timestamp;
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(DO_TIMER);
#endif
  objID = new state_worker(m);
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(SIM_TIMER);
#endif
  myStrat->init(eq, objID, this, thisIndex);
#ifdef POSE_STATS_ON
  localStats->TimerStop();
#endif
#ifndef SEQUENTIAL_POSE
  PVT *pvt = (PVT *)CkLocalBranch(ThePVT);
  myPVTidx = pvt->objRegister(thisIndex, _ts, sync, this);
#endif
#ifdef LB_ON  
  myLBidx = localLBG->objRegister(thisIndex, sync, this);
#endif  
}
void worker::pup(PUP::er &p)
  {
    sim::pup(p);
    if (p.isUnpacking()) {
      myStrat = new adapt3;
      objID = new state_worker(this, myStrat);
      myStrat->init(eq, objID, this, thisIndex);
    }
    ((state_worker *)objID)->pup(p);
    Event *ev = eq->front()->next;
    int checkpointed;

    while (ev != eq->back()) {
      if (p.isUnpacking()) {
        p(checkpointed);
        if (checkpointed) {
          ev->cpData = new state_worker(this, myStrat);
          ((state_worker *)ev->cpData)->cpPup(p);
        }
        else ev->cpData = NULL;
      }
      else {
        if (ev->cpData) {
          checkpointed = 1;
          p(checkpointed);
          ((state_worker *)ev->cpData)->cpPup(p);
        }
        else {
          checkpointed = 0;
          p(checkpointed);
        }
      } 
     ev=ev->next; 
    }
  }
state_worker::state_worker(WorkerData *m){
init(m);
  int i;
  numObjs = m->numObjs;
  numMsgs = m->numMsgs;
  msgSize = m->msgSize;
  locality = m->locality;
  grainSize = m->grainSize;
  granularity = m->granularity;
  density = m->density;
  msgsPerWork = m->msgsPerWork;
  for (i=0; i<100; i++) data[i] = 0;
  sent = 0;
  totalObjs = numObjs * CkNumPes();
  localDensity = ((double)density)/((double)totalObjs);
  elapseTime = (int)(1.0/localDensity);
  elapseRem = (int)(((1.0/localDensity) * (double)msgsPerWork) - 		    (elapseTime * msgsPerWork));
  neighbor = (myHandle + numObjs) % totalObjs;
  delete m;
  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
    
{
int _POSE_handle = parent->thisIndex;
POSE_TimeType _POSE_timeOffset =  0;
CkAssert(_POSE_timeOffset >=0);sm->Timestamp(ovt+(_POSE_timeOffset));
#ifndef SEQUENTIAL_POSE
PVT *pvt = (PVT *)CkLocalBranch(ThePVT);
pvt->objUpdate(ovt+(_POSE_timeOffset), SEND);
#endif
#ifndef CMK_OPTIMIZE
sm->sanitize();
#endif
(* (CProxy_worker *)&POSE_Objects)[_POSE_handle].workSmall(sm);
int _destPE = POSE_Objects.ckLocalBranch()->lastKnown(CkArrayIndex1D(_POSE_handle));
parent->srVector[_destPE]++;
}

  sent++;
}
state_worker::state_worker(){
}
state_worker& state_worker::operator=(const state_worker& obj){
  int i;
  rep::operator=(obj);
  numObjs = obj.numObjs;
  numMsgs = obj.numMsgs;
  msgSize = obj.msgSize;
  locality = obj.locality;
  grainSize = obj.grainSize;
  granularity = obj.granularity;
  density = obj.density;
  msgsPerWork = obj.msgsPerWork;
  sent = obj.sent;
  totalObjs = obj.totalObjs;
  elapseTime = obj.elapseTime;
  elapseRem = obj.elapseRem;
  neighbor = obj.neighbor;
  for (i=0; i<100; i++) data[i] = obj.data[i];
  return *this;
}
void worker::workSmall(SmallWorkMsg *m){
#ifndef CMK_OPTIMIZE
m->sanitize();
#endif
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (tstat)
    localStats->SwitchTimer(SIM_TIMER);
  else
    localStats->TimerStart(SIM_TIMER);
#endif
#ifndef SEQUENTIAL_POSE
  PVT *pvt = (PVT *)CkLocalBranch(ThePVT);
  pvt->objUpdate(m->timestamp, RECV);
  srVector[m->evID.getPE()]++;
#endif
  Event *e = new Event();
  if ((POSE_endtime < 0) || (m->timestamp <= POSE_endtime)) {
    e->evID = m->evID;
    e->timestamp = m->timestamp;
    e->done = e->commitBfrLen = 0;
    e->commitBfr = NULL;
    e->msg = m;
    e->fnIdx = 1;
    e->next = e->prev = NULL;
    e->spawnedList = NULL;
#ifndef SEQUENTIAL_POSE
    CkAssert(e->timestamp >= pvt->getGVT());
#endif
    eq->InsertEvent(e);
    Step();
  }
#ifdef POSE_STATS_ON
  if (tstat)
    localStats->SwitchTimer(tstat);
  else
    localStats->TimerStop();
#endif
}
void state_worker::workSmall(SmallWorkMsg *m){
  //CkPrintf("%d receiving small work at %d\n", parent->thisIndex, ovt);
  doWork();
}
void state_worker::workSmall_anti(SmallWorkMsg *m){
  restore(this);
}
void state_worker::workSmall_commit(SmallWorkMsg *m){
}
void worker::workMedium(MediumWorkMsg *m){
#ifndef CMK_OPTIMIZE
m->sanitize();
#endif
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (tstat)
    localStats->SwitchTimer(SIM_TIMER);
  else
    localStats->TimerStart(SIM_TIMER);
#endif
#ifndef SEQUENTIAL_POSE
  PVT *pvt = (PVT *)CkLocalBranch(ThePVT);
  pvt->objUpdate(m->timestamp, RECV);
  srVector[m->evID.getPE()]++;
#endif
  Event *e = new Event();
  if ((POSE_endtime < 0) || (m->timestamp <= POSE_endtime)) {
    e->evID = m->evID;
    e->timestamp = m->timestamp;
    e->done = e->commitBfrLen = 0;
    e->commitBfr = NULL;
    e->msg = m;
    e->fnIdx = 2;
    e->next = e->prev = NULL;
    e->spawnedList = NULL;
#ifndef SEQUENTIAL_POSE
    CkAssert(e->timestamp >= pvt->getGVT());
#endif
    eq->InsertEvent(e);
    Step();
  }
#ifdef POSE_STATS_ON
  if (tstat)
    localStats->SwitchTimer(tstat);
  else
    localStats->TimerStop();
#endif
}
void state_worker::workMedium(MediumWorkMsg *m){
  //CkPrintf("%d receiving medium work at %d\n", parent->thisIndex, ovt);
  doWork();
}
void state_worker::workMedium_anti(MediumWorkMsg *m){
  restore(this);
}
void state_worker::workMedium_commit(MediumWorkMsg *m){
}
void worker::workLarge(LargeWorkMsg *m){
#ifndef CMK_OPTIMIZE
m->sanitize();
#endif
#ifdef POSE_STATS_ON
  int tstat = localStats->TimerRunning();
  if (tstat)
    localStats->SwitchTimer(SIM_TIMER);
  else
    localStats->TimerStart(SIM_TIMER);
#endif
#ifndef SEQUENTIAL_POSE
  PVT *pvt = (PVT *)CkLocalBranch(ThePVT);
  pvt->objUpdate(m->timestamp, RECV);
  srVector[m->evID.getPE()]++;
#endif
  Event *e = new Event();
  if ((POSE_endtime < 0) || (m->timestamp <= POSE_endtime)) {
    e->evID = m->evID;
    e->timestamp = m->timestamp;
    e->done = e->commitBfrLen = 0;
    e->commitBfr = NULL;
    e->msg = m;
    e->fnIdx = 3;
    e->next = e->prev = NULL;
    e->spawnedList = NULL;
#ifndef SEQUENTIAL_POSE
    CkAssert(e->timestamp >= pvt->getGVT());
#endif
    eq->InsertEvent(e);
    Step();
  }
#ifdef POSE_STATS_ON
  if (tstat)
    localStats->SwitchTimer(tstat);
  else
    localStats->TimerStop();
#endif
}
void state_worker::workLarge(LargeWorkMsg *m){
  //CkPrintf("%d receiving large work at %d\n", parent->thisIndex, ovt);
  doWork();
}
void state_worker::workLarge_anti(LargeWorkMsg *m){
  restore(this);
}
void state_worker::workLarge_commit(LargeWorkMsg *m){
}
void state_worker::doWork(){
  SmallWorkMsg *sm;
  MediumWorkMsg *mm;
  LargeWorkMsg *lm;
  if ((POSE_endtime > -1) && (OVT() > POSE_endtime))  return;
  // do some computation based on gsIdx
  if (granularity > 0.0) POSE_busy_wait(granularity);
  else if (grainSize == FINE) POSE_busy_wait(FINE_GRAIN);
  else if (grainSize == MEDIUM_GS) POSE_busy_wait(MEDIUM_GRAIN);
  else if (grainSize == COARSE) POSE_busy_wait(COARSE_GRAIN);
  else if (grainSize == MIX_GS) POSE_busy_wait(MEDIUM_GRAIN);
  // generate some events
  int actualMsgSize = msgSize;
  int local = (int)(((double)locality)/100.0 * (double)msgsPerWork);
  int localNbr = (myHandle+1) % numObjs;
  for (int i=0; i<msgsPerWork; i++) {
    if (sent >= numMsgs) return;
    elapse(elapseTime);
    if (msgSize == MIX_MS) actualMsgSize = (actualMsgSize + 1) % 3;
    if (actualMsgSize == SMALL) {
      sm = new SmallWorkMsg;
      memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
      if (local > 0) local--;
      else localNbr = neighbor;
      
{
if (!CpvAccess(stateRecovery)) {
int _POSE_handle = localNbr;
POSE_TimeType _POSE_timeOffset =  0;
registerTimestamp(_POSE_handle, sm,_POSE_timeOffset);
#ifdef POSE_DOP_ON
parent->ct = CmiWallTimer();
sm->rst = parent->ct - parent->st + parent->eq->currentPtr->srt;
#endif
#ifndef CMK_OPTIMIZE
sm->sanitize();
#endif
(* (CProxy_worker *)&POSE_Objects)[_POSE_handle].workSmall(sm);
int _destPE = POSE_Objects.ckLocalBranch()->lastKnown(CkArrayIndex1D(_POSE_handle));
parent->srVector[_destPE]++;
}
else delete sm;}

      //CkPrintf("%d sending small work to %d at %d. Sent=%d\n",myHandle,localNbr,ovt,sent);
    }
    else if (actualMsgSize == MEDIUM) {
      mm = new MediumWorkMsg;
      memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
      if (local > 0) local--;
      else localNbr = neighbor;
      
{
if (!CpvAccess(stateRecovery)) {
int _POSE_handle = localNbr;
POSE_TimeType _POSE_timeOffset =  0;
registerTimestamp(_POSE_handle, mm,_POSE_timeOffset);
#ifdef POSE_DOP_ON
parent->ct = CmiWallTimer();
mm->rst = parent->ct - parent->st + parent->eq->currentPtr->srt;
#endif
#ifndef CMK_OPTIMIZE
mm->sanitize();
#endif
(* (CProxy_worker *)&POSE_Objects)[_POSE_handle].workMedium(mm);
int _destPE = POSE_Objects.ckLocalBranch()->lastKnown(CkArrayIndex1D(_POSE_handle));
parent->srVector[_destPE]++;
}
else delete mm;}

      //CkPrintf("%d sending medium work to %d at %d\n",myHandle,localNbr,ovt);
    }
    else if (actualMsgSize == LARGE) {
      lm = new LargeWorkMsg;
      memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
      if (local > 0) local--;
      else localNbr = neighbor;
      
{
if (!CpvAccess(stateRecovery)) {
int _POSE_handle = localNbr;
POSE_TimeType _POSE_timeOffset =  0;
registerTimestamp(_POSE_handle, lm,_POSE_timeOffset);
#ifdef POSE_DOP_ON
parent->ct = CmiWallTimer();
lm->rst = parent->ct - parent->st + parent->eq->currentPtr->srt;
#endif
#ifndef CMK_OPTIMIZE
lm->sanitize();
#endif
(* (CProxy_worker *)&POSE_Objects)[_POSE_handle].workLarge(lm);
int _destPE = POSE_Objects.ckLocalBranch()->lastKnown(CkArrayIndex1D(_POSE_handle));
parent->srVector[_destPE]++;
}
else delete lm;}

      //CkPrintf("%d sending large work to %d at %d\n",myHandle,localNbr,ovt);
    }
    sent++;
  }
  elapse(elapseRem);
  int elapseCheck = sent * (1.0/density);
  if (OVT() < elapseCheck) elapse(elapseCheck-OVT());
}

void worker::ResolveFn(int fnIdx, void *msg)
{
  if (fnIdx >0){
    if (sync == OPTIMISTIC)
      ((state_worker *) objID)->checkpoint((state_worker *) objID);
    ((state_worker *) objID)->update(((eventMsg *)msg)->timestamp, ((eventMsg *)msg)->rst);
  }
  if (fnIdx == 1) {
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {localStats->Do();
#ifdef POSE_DOP_ON
    st = CmiWallTimer();
#endif
    localStats->SwitchTimer(DO_TIMER);}
#endif
    ((state_worker *) objID)->workSmall((SmallWorkMsg *)msg);
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {
#ifdef POSE_DOP_ON
    et = CmiWallTimer();
    eq->currentPtr->ert = eq->currentPtr->srt + (et-st);
    ((state_worker *) objID)->ort = eq->currentPtr->ert+0.000001;
    eq->currentPtr->evt = ((state_worker *) objID)->OVT();
#endif
    localStats->SwitchTimer(SIM_TIMER);}
#endif
  }
  else if (fnIdx == -1) {
    ((state_worker *) objID)->workSmall_anti((SmallWorkMsg *)msg);
  }
  else if (fnIdx == 2) {
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {localStats->Do();
#ifdef POSE_DOP_ON
    st = CmiWallTimer();
#endif
    localStats->SwitchTimer(DO_TIMER);}
#endif
    ((state_worker *) objID)->workMedium((MediumWorkMsg *)msg);
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {
#ifdef POSE_DOP_ON
    et = CmiWallTimer();
    eq->currentPtr->ert = eq->currentPtr->srt + (et-st);
    ((state_worker *) objID)->ort = eq->currentPtr->ert+0.000001;
    eq->currentPtr->evt = ((state_worker *) objID)->OVT();
#endif
    localStats->SwitchTimer(SIM_TIMER);}
#endif
  }
  else if (fnIdx == -2) {
    ((state_worker *) objID)->workMedium_anti((MediumWorkMsg *)msg);
  }
  else if (fnIdx == 3) {
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {localStats->Do();
#ifdef POSE_DOP_ON
    st = CmiWallTimer();
#endif
    localStats->SwitchTimer(DO_TIMER);}
#endif
    ((state_worker *) objID)->workLarge((LargeWorkMsg *)msg);
#ifdef POSE_STATS_ON
    if (!CpvAccess(stateRecovery)) {
#ifdef POSE_DOP_ON
    et = CmiWallTimer();
    eq->currentPtr->ert = eq->currentPtr->srt + (et-st);
    ((state_worker *) objID)->ort = eq->currentPtr->ert+0.000001;
    eq->currentPtr->evt = ((state_worker *) objID)->OVT();
#endif
    localStats->SwitchTimer(SIM_TIMER);}
#endif
  }
  else if (fnIdx == -3) {
    ((state_worker *) objID)->workLarge_anti((LargeWorkMsg *)msg);
  }
}

void worker::ResolveCommitFn(int fnIdx, void *msg)
{
  if (fnIdx == 1) {
    ((state_worker *) objID)->workSmall_commit((SmallWorkMsg *)msg);
  }
  else if (fnIdx == 2) {
    ((state_worker *) objID)->workMedium_commit((MediumWorkMsg *)msg);
  }
  else if (fnIdx == 3) {
    ((state_worker *) objID)->workLarge_commit((LargeWorkMsg *)msg);
  }
}

