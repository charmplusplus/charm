#include <math.h>

int GCD(int a, int b) {
  if (b > a) 
    return GCD(b,a);
  else if(b == 0) 
    return a;
  else
    return GCD(b, a%b);
}


worker::worker(WorkerData *m)
{
  int i;
  numObjs = m->numObjs;
  numMsgs = m->numMsgs;
  msgSize = m->msgSize;
  grainSize = m->grainSize;
  granularity = m->granularity;
  density = m->density;
  for (i=0; i<100; i++) data[i] = 0;
  sent = 0;
  totalObjs = numObjs * CkNumPes();
  localDensity = ((double)density)/((double)totalObjs);
  delete m;

  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
  sm->fromPE = -1;
  //CkPrintf("Worker %d created on PE %d.\n", myHandle, CkMyPe());
  //if (myHandle%numObjs == 0) { //local ring; multiple global rings
  //if (myHandle%(numObjs/2) == 0) { //multiple offset global rings
  //if (myHandle == 0) { 
  //CkPrintf("Worker %d starting ring, sending to self.\n", myHandle);
  POSE_invoke(workSmall(sm), worker, parent->thisIndex, 0);
  //}
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
  grainSize = obj.grainSize;
  granularity = obj.granularity;
  density = obj.density;
  sent = obj.sent;
  totalObjs = obj.totalObjs;
  for (i=0; i<100; i++) data[i] = obj.data[i];
  return *this;
}

void worker::terminus()
{
  if (sent != numMsgs)
    CkPrintf("%d sent %d messages!\n", myHandle, sent);
}

void worker::workSmall(SmallWorkMsg *m)
{
  //CkPrintf("%d receiving small work at %d from obj %d\n", parent->thisIndex, ovt, m->fromPE);
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
  int nbr;

  if (sent == numMsgs) return;
  sent++;

  // generate an event
  int actualMsgSize = msgSize;
  if (msgSize == MIX_MS) actualMsgSize = (actualMsgSize + 1) % 3;
  if (actualMsgSize == SMALL) {
    sm = new SmallWorkMsg;
    memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
    sm->fromPE = myHandle;
    // local ring
    POSE_invoke(workSmall(sm), worker, ((myHandle%numObjs)+1)%numObjs + (numObjs*CkMyPe()), 0);
    // global ring
    //POSE_invoke(workSmall(sm), worker, (myHandle+1)%totalObjs, 0);
    //CkPrintf("%d sending small work to %d at %d. Sent=%d\n",myHandle,((myHandle%numObjs)+1)%numObjs + (numObjs*CkMyPe()),ovt,sent);
  }
  else if (actualMsgSize == MEDIUM) {
    mm = new MediumWorkMsg;
    memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
    POSE_invoke(workMedium(mm), worker, (myHandle+1)%totalObjs, 0);
    //CkPrintf("%d sending medium work to %d at %d\n",myHandle,nbr,ovt);
  }
  else if (actualMsgSize == LARGE) {
    lm = new LargeWorkMsg;
    memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
    POSE_invoke(workLarge(lm), worker, (myHandle+1)%totalObjs, 0);
    //CkPrintf("%d sending large work to %d at %d\n",myHandle,nbr,ovt);
  }
  int elapseCheck = sent * (1.0/localDensity);
  if (OVT() < elapseCheck) elapse(elapseCheck-OVT());
  //CkPrintf("%d sent %d messages out of %d!\n", myHandle, sent, numMsgs);
}

