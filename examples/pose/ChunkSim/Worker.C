#include <math.h>

void worker::set(WorkerData *m)
{
  int i;
  workerID = m->workerID;
  numWorkers = m->numWorkers;
  numObjs = m->numObjs;
  numMsgs = m->numMsgs;
  msgSize = m->msgSize;
  distribution = m->distribution;
  connectivity = m->connectivity;
  locality = m->locality;
  offsetPattern = m->offsetPattern;
  sendPattern = m->sendPattern;
  for (i=0; i<100; i++) data[i] = 0;
  for (i=0; i<5; i++) {
    numSends[i] = m->numSends[i];
    offsets[i] = m->offsets[i];
  } 
  numNbrs = m->numNbrs;
  for (i=0; i<numNbrs; i++) neighbors[i] = m->neighbors[i];
  sendIdx = nbrIdx = offsetIdx = msgIdx = 0;
}

worker::worker()
{
  sendIdx = nbrIdx = offsetIdx = msgIdx = 0;
}

worker& worker::operator=(const worker& obj)
{
  int i;
  workerID = obj.workerID;
  numWorkers = obj.numWorkers;
  numObjs = obj.numObjs;
  numMsgs = obj.numMsgs;
  msgSize = obj.msgSize;
  distribution = obj.distribution;
  connectivity = obj.connectivity;
  locality = obj.locality;
  offsetPattern = obj.offsetPattern;
  sendPattern = obj.sendPattern;
  for (i=0; i<100; i++) data[i] = obj.data[i];
  for (i=0; i<5; i++) {
    numSends[i] = obj.numSends[i];
    offsets[i] = obj.offsets[i];
  }
  for (i=0; i<100; i++) neighbors[i] = obj.neighbors[i];
  numNbrs = obj.numNbrs;
  sendIdx = obj.sendIdx;
  nbrIdx = obj.nbrIdx;
  offsetIdx = obj.offsetIdx;
  msgIdx = obj.msgIdx;
  return *this;
}


team::team(TeamData *m)
{
  workersRecvd = 0;
  numTeams = m->numTeams;
  numWorkers = m->numWorkers;
  numObjs = m->numObjs;
  myWorkers = new worker[numWorkers];
}

void team::addWorker(WorkerData *wd) 
{
  eventMsg *em;
  myWorkers[wd->workerID%numWorkers].set(wd);
  CkFreeMsg(wd);
  workersRecvd++;
  if ((parent->thisIndex == numTeams-1) && (workersRecvd == numWorkers))
    for (int i=0; i<numTeams; i++) {
      em = new eventMsg;
      POSE_invoke(start(em), team, i, 0);
    }
}

void team::start(eventMsg *em)
{
  SmallWorkMsg *sm;

  for (int i=0; i<numWorkers; i++) {
    sm = new SmallWorkMsg;
    sm->workerID = myWorkers[i].workerID;
    memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
    POSE_local_invoke(workSmall(sm), 0);
  }
}
void team::start_anti(eventMsg *em) {restore(this);}
void team::start_commit(eventMsg *em) {}


void team::workSmall(SmallWorkMsg *sm) 
{
  doWork(sm->workerID%numWorkers);
}
void team::workSmall_anti(SmallWorkMsg *sm) {restore(this);}
void team::workSmall_commit(SmallWorkMsg *sm) {}

void team::workMedium(MediumWorkMsg *mm) 
{
  doWork(mm->workerID%numWorkers);
}
void team::workMedium_anti(MediumWorkMsg *mm) {restore(this);}
void team::workMedium_commit(MediumWorkMsg *mm) {}

void team::workLarge(LargeWorkMsg *lm) 
{
  doWork(lm->workerID%numWorkers);
}
void team::workLarge_anti(LargeWorkMsg *lm) {restore(this);}
void team::workLarge_commit(LargeWorkMsg *lm) {}

void team::doWork(int k)
{
  SmallWorkMsg *sm;
  MediumWorkMsg *mm;
  LargeWorkMsg *lm;
  
  if ((POSE_endtime > -1) && (OVT() > POSE_endtime))  return;

  // generate some events
  int actualMsgSize = myWorkers[k].msgSize;
  for (int i=0; i<myWorkers[k].numSends[myWorkers[k].sendIdx]; i++) {
    if (myWorkers[k].msgSize == MIX_MS) {
      actualMsgSize = myWorkers[k].msgIdx+1;
      myWorkers[k].msgIdx = (myWorkers[k].msgIdx + 1) % 3;
    }
    
    if (actualMsgSize == SMALL) {
      sm = new SmallWorkMsg;
      sm->workerID = myWorkers[k].neighbors[myWorkers[k].nbrIdx];
      memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
      POSE_invoke(workSmall(sm), team, myWorkers[k].neighbors[myWorkers[k].nbrIdx]/numWorkers, myWorkers[k].offsets[myWorkers[k].offsetIdx]);
    }
    else if (actualMsgSize == MEDIUM) {
      mm = new MediumWorkMsg;
      mm->workerID = myWorkers[k].neighbors[myWorkers[k].nbrIdx];
      memset(mm->data, 0, MD_MSG_SZ*sizeof(int));
      POSE_invoke(workMedium(mm), team, myWorkers[k].neighbors[myWorkers[k].nbrIdx]/numWorkers, myWorkers[k].offsets[myWorkers[k].offsetIdx]);
    }
    else if (actualMsgSize == LARGE) {
      lm = new LargeWorkMsg;
      lm->workerID = myWorkers[k].neighbors[myWorkers[k].nbrIdx];
      memset(lm->data, 0, LG_MSG_SZ*sizeof(int));
      POSE_invoke(workLarge(lm), team, myWorkers[k].neighbors[myWorkers[k].nbrIdx]/numWorkers, myWorkers[k].offsets[myWorkers[k].offsetIdx]);
    }
    myWorkers[k].nbrIdx = (myWorkers[k].nbrIdx + 1) % myWorkers[k].numNbrs;
    myWorkers[k].offsetIdx = (myWorkers[k].offsetIdx + 1) % 5;
  }
  myWorkers[k].sendIdx = (myWorkers[k].sendIdx + 1) % 5;
}

