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
  locality = m->locality;
  grainSize = m->grainSize;
  granularity = m->granularity;
  density = m->density;
  for (i=0; i<100; i++) data[i] = 0;
  sent = 0;
  totalObjs = numObjs * CkNumPes();
  localDensity = ((double)density)/((double)totalObjs);
  localMsgs = locality;
  remoteMsgs = 100 - locality;
  int theGCD = GCD(localMsgs, remoteMsgs);
  localMsgs = localMsgs/theGCD;
  remoteMsgs = remoteMsgs/theGCD;
  localCount = localMsgs;
  remoteCount = remoteMsgs;
  localNbr = ((myHandle+1) % numObjs) + (CkMyPe() * numObjs);
  remoteNbr = (myHandle + numObjs) % totalObjs;
  fromLocal = ((myHandle + numObjs -1) % numObjs) + (CkMyPe() * numObjs);
  fromRemote = (myHandle + totalObjs - numObjs) % totalObjs;
  delete m;
  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
  sm->fromPE = -1;
  //CkPrintf("Worker %d created on PE %d with %d/%d (local/remote messages) sends to %d/%d receives from %d/%d.\n", myHandle, CkMyPe(), localMsgs, remoteMsgs, localNbr, remoteNbr, fromLocal, fromRemote);
  POSE_invoke(workSmall(sm), worker, parent->thisIndex, 0);
  sent++;
  received = 0;
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
  sent = obj.sent;
  received = obj.received;
  localMsgs = obj.localMsgs;
  remoteMsgs = obj.remoteMsgs;
  localCount = obj.localCount;
  remoteCount = obj.remoteCount;
  localNbr = obj.localNbr;
  remoteNbr = obj.remoteNbr;
  fromLocal = obj.fromLocal;
  fromRemote = obj.fromRemote;
  totalObjs = obj.totalObjs;
  for (i=0; i<100; i++) data[i] = obj.data[i];
  return *this;
}

void worker::terminus()
{
  if (sent != numMsgs)
    CkPrintf("%d sent %d messages!\n", myHandle, sent);
  if (received != numMsgs)
    CkPrintf("%d received %d messages!\n", myHandle, received);
}

void worker::workSmall(SmallWorkMsg *m)
{
  //CkPrintf("%d receiving small work at %d\n", parent->thisIndex, ovt);
  received++;
  if ((m->fromPE != fromLocal) && (m->fromPE != fromRemote) && (m->fromPE!=-1))
    parent->CommitPrintf("%d received from %d which is weird!\n", myHandle, m->fromPE);
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

  if (sent > numMsgs) parent->CommitPrintf("%d received more msgs than sent!\n", myHandle);
  if (sent >= numMsgs) return;
  sent++;
  if (localCount > 0) {
    localCount--;
    nbr = localNbr;
  }
  else if (remoteCount > 0) {
    remoteCount--;
    nbr = remoteNbr;
    if (remoteCount == 0) {
      localCount = localMsgs;
      remoteCount = remoteMsgs;
    }
  }
  // generate an event
  int actualMsgSize = msgSize;
  if (msgSize == MIX_MS) actualMsgSize = (actualMsgSize + 1) % 3;
  if (actualMsgSize == SMALL) {
    sm = new SmallWorkMsg;
    memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
    sm->fromPE = myHandle;
    POSE_invoke(workSmall(sm), worker, nbr, 0);
    //CkPrintf("%d sending small work to %d at %d. Sent=%d\n",myHandle,nbr,ovt,sent);
  }
  else if (actualMsgSize == MEDIUM) {
    mm = new MediumWorkMsg;
    memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
    POSE_invoke(workMedium(mm), worker, nbr, 0);
    //CkPrintf("%d sending medium work to %d at %d\n",myHandle,nbr,ovt);
  }
  else if (actualMsgSize == LARGE) {
    lm = new LargeWorkMsg;
    memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
    POSE_invoke(workLarge(lm), worker, nbr, 0);
    //CkPrintf("%d sending large work to %d at %d\n",myHandle,nbr,ovt);
  }
  int elapseCheck = sent * (1.0/localDensity);
  if (OVT() < elapseCheck) elapse(elapseCheck-OVT());
  //  if (sent == numMsgs)
  //    CkPrintf("%d sent %d & received %d messages!\n", myHandle, sent, received);
}

