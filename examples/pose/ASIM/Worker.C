#include <math.h>

worker::worker(WorkerData *m)
{
  int i;
  numObjs = m->numObjs;
  numMsgs = m->numMsgs;
  msgSize = m->msgSize;
  distribution = m->distribution;
  connectivity = m->connectivity;
  locality = m->locality;
  grainSize = m->grainSize;
  elapsePattern = m->elapsePattern;
  offsetPattern = m->offsetPattern;
  sendPattern = m->sendPattern;
  granularity = m->granularity;
  for (i=0; i<100; i++) data[i] = 0;
  for (i=0; i<5; i++) {
    elapseTimes[i] = m->elapseTimes[i];
    numSends[i] = m->numSends[i];
    offsets[i] = m->offsets[i];
  } 
  numNbrs = m->numNbrs;
  for (i=0; i<numNbrs; i++) neighbors[i] = m->neighbors[i];
  delete m;
  elapseIdx = sendIdx = nbrIdx = offsetIdx = msgIdx = gsIdx = 0;
  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
  //CkPrintf("Worker %d created on PE %d. Sending message to self...\n", parent->thisIndex, CkMyPe());
  POSE_invoke(workSmall(sm), worker, parent->thisIndex, 0);
}

worker::worker()
{
  elapseIdx = sendIdx = nbrIdx = offsetIdx = msgIdx = gsIdx = 0;
}

worker& worker::operator=(const worker& obj)
{
  int i;
  rep::operator=(obj);
  numObjs = obj.numObjs;
  numMsgs = obj.numMsgs;
  msgSize = obj.msgSize;
  distribution = obj.distribution;
  connectivity = obj.connectivity;
  locality = obj.locality;
  grainSize = obj.grainSize;
  elapsePattern = obj.elapsePattern;
  offsetPattern = obj.offsetPattern;
  sendPattern = obj.sendPattern;
  granularity = obj.granularity;
  for (i=0; i<100; i++) data[i] = obj.data[i];
  for (i=0; i<5; i++) {
    elapseTimes[i] = obj.elapseTimes[i];
    numSends[i] = obj.numSends[i];
    offsets[i] = obj.offsets[i];
  }
  for (i=0; i<100; i++) neighbors[i] = obj.neighbors[i];
  numNbrs = obj.numNbrs;
  elapseIdx = obj.elapseIdx;
  sendIdx = obj.sendIdx;
  nbrIdx = obj.nbrIdx;
  offsetIdx = obj.offsetIdx;
  msgIdx = obj.msgIdx;
  gsIdx = obj.gsIdx;
  return *this;
}


void worker::workSmall(SmallWorkMsg *m)
{
  //CkPrintf("%d receiving small work at %d\n", parent->thisIndex, ovt);
  doWork();
}

void worker::workSmall_anti(SmallWorkMsg *m)
{
  restore(this);
}

void worker::workSmall_commit(SmallWorkMsg *m)
{
}

void worker::workMedium(MediumWorkMsg *m)
{
  //CkPrintf("%d receiving medium work at %d\n", parent->thisIndex, ovt);
  doWork();
}

void worker::workMedium_anti(MediumWorkMsg *m)
{
  restore(this);
}

void worker::workMedium_commit(MediumWorkMsg *m)
{
}

void worker::workLarge(LargeWorkMsg *m)
{
  //CkPrintf("%d receiving large work at %d\n", parent->thisIndex, ovt);
  doWork();
}

void worker::workLarge_anti(LargeWorkMsg *m)
{
  restore(this);
}

void worker::workLarge_commit(LargeWorkMsg *m)
{
}

void worker::doWork()
{
  SmallWorkMsg *sm;
  MediumWorkMsg *mm;
  LargeWorkMsg *lm;

  if ((POSE_endtime > -1) && (OVT() > POSE_endtime))  return;

  // do some computation based on gsIdx
  if (granularity > 0.0) POSE_busy_wait(granularity);
  else if (grainSize == FINE) POSE_busy_wait(FINE_GRAIN);
  else if (grainSize == MEDIUM_GS) POSE_busy_wait(MEDIUM_GRAIN);
  else if (grainSize == COARSE) POSE_busy_wait(COARSE_GRAIN);
  else if (grainSize == MIX_GS) {
    if (gsIdx == FINE) POSE_busy_wait(FINE_GRAIN);
    else if (gsIdx == MEDIUM_GS) POSE_busy_wait(MEDIUM_GRAIN);
    else if (gsIdx == COARSE) POSE_busy_wait(COARSE_GRAIN);
    gsIdx = (gsIdx + 1) % 3;
  }

  // elapse some time
  elapse(elapseTimes[elapseIdx]);
  elapseIdx = (elapseIdx + 1) % 5;

  // generate some events
  int actualMsgSize = msgSize;
  for (int i=0; i<numSends[sendIdx]; i++) {
    if (msgSize == MIX_MS) {
      actualMsgSize = msgIdx+1;
      msgIdx = (msgIdx + 1) % 3;
    }
    
    if (actualMsgSize == SMALL) {
      sm = new SmallWorkMsg;
      memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
      //CkPrintf("%d sending small work to %d at %d + %d\n", parent->thisIndex,
      //       neighbors[nbrIdx], ovt, offsets[offsetIdx]);
      POSE_invoke(workSmall(sm), worker, neighbors[nbrIdx], offsets[offsetIdx]);
    }
    else if (actualMsgSize == MEDIUM) {
      mm = new MediumWorkMsg;
      memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
      //CkPrintf("%d sending medium work to %d at %d + %d\n", parent->thisIndex, 
      //       neighbors[nbrIdx], ovt, offsets[offsetIdx]);
      POSE_invoke(workMedium(mm), worker, neighbors[nbrIdx], offsets[offsetIdx]);
    }
    else if (actualMsgSize == LARGE) {
      lm = new LargeWorkMsg;
      memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
      //CkPrintf("%d sending large work to %d at %d + %d\n", parent->thisIndex,
      //       neighbors[nbrIdx], ovt, offsets[offsetIdx]);
      POSE_invoke(workLarge(lm), worker, neighbors[nbrIdx], offsets[offsetIdx]);
    }
    nbrIdx = (nbrIdx + 1) % numNbrs;
    offsetIdx = (offsetIdx + 1) % 5;
  }
  sendIdx = (sendIdx + 1) % 5;
}

