#include <math.h>

worker::worker(WorkerData *m)
{
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
  elapseRem = (int)(((1.0/localDensity) * (double)msgsPerWork) - 
		    (elapseTime * msgsPerWork));
  neighbor = (myHandle + numObjs) % totalObjs;
  delete m;
  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
  //CkPrintf("Worker %d created on PE %d with msgsPerWork=%d, elapseTime=%d and elapseRem=%d. Sending message to self...\n", myHandle, CkMyPe(), msgsPerWork, elapseTime, elapseRem);
  POSE_invoke(workSmall(sm), worker, parent->thisIndex, 0);
  sent++;
}

worker::worker()
{
}

worker& worker::operator=(const worker& obj)
{
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

  // generate some events
  int actualMsgSize = msgSize;
  int local = (int)(((double)locality)/100.0 * (double)msgsPerWork);
  //CkPrintf("%d out of %d messages will be sent to local objects\n", local, msgsPerWork);
  int localNbr;
  for (int i=0; i<msgsPerWork; i++) {
    if (sent >= numMsgs) return;
    sent++;
    localNbr = ((myHandle+1) % numObjs) + (CkMyPe() * numObjs);
    if (local > 0) local--;
    else localNbr = neighbor;
    elapse(elapseTime);
    if (msgSize == MIX_MS) actualMsgSize = (actualMsgSize + 1) % 3;
    if (actualMsgSize == SMALL) {
      sm = new SmallWorkMsg;
      memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
      POSE_invoke(workSmall(sm), worker, localNbr, 0);
      //CkPrintf("%d sending small work to %d at %d. Sent=%d\n",myHandle,localNbr,ovt,sent);
    }
    else if (actualMsgSize == MEDIUM) {
      mm = new MediumWorkMsg;
      memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
      POSE_invoke(workMedium(mm), worker, localNbr, 0);
      //CkPrintf("%d sending medium work to %d at %d\n",myHandle,localNbr,ovt);
    }
    else if (actualMsgSize == LARGE) {
      lm = new LargeWorkMsg;
      memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
      POSE_invoke(workLarge(lm), worker, localNbr, 0);
      //CkPrintf("%d sending large work to %d at %d\n",myHandle,localNbr,ovt);
    }
  }
  elapse(elapseRem);
  int elapseCheck = sent * (1.0/density);
  if (OVT() < elapseCheck) elapse(elapseCheck-OVT());
}

