#include <math.h>

void worker::set(int wid)
{
  int i;
  workerID = wid;
  for (i=0; i<WORKER_SZ; i++) data[i] = 0;
}

worker::worker() { for (int i=0; i<WORKER_SZ; i++) data[i] = 0; }

worker& worker::operator=(const worker& obj)
{
  int i;
  workerID = obj.workerID;
  for (i=0; i<WORKER_SZ; i++) data[i] = obj.data[i];
  return *this;
}


team::team(TeamData *m)
{
  teamID = m->teamID;
  numTeams = m->numTeams;
  numWorkers = m->numWorkers;
  myWorkers = new worker[numWorkers/numTeams];

  WorkMsg *wm;
  int offset = teamID * (numWorkers/numTeams);
  CkAssert(offset >= 0);
  //CkPrintf("Team %d(%d) constructed.  Offset=%d\n", parent->thisIndex, teamID, offset);
  for (int i=0; i<numWorkers/numTeams; i++) {
    myWorkers[i].set(offset+i);
    wm = new WorkMsg;
    wm->workerID = offset+i;
    memset(wm->data, 0, 10*sizeof(int));
    //CkPrintf("Team %d(%d) generated initial work for worker %d\n", parent->thisIndex, teamID, wm->workerID);
    POSE_local_invoke(work(wm), 0);
  }
}

void team::work(WorkMsg *wm) 
{
  doWork(wm->workerID);
}
void team::work_anti(WorkMsg *wm) {restore(this);}
void team::work_commit(WorkMsg *wm) {}

void team::doWork(int k)
{
  WorkMsg *wm;
  
  if ((POSE_endtime > -1) && (OVT() > POSE_endtime))  return;
  for (int j=0; j<WORKER_SZ; j++) {
    myWorkers[k%(numWorkers/numTeams)].data[j] += 
      myWorkers[k%(numWorkers/numTeams)].data[99-j];
  } 
  // generate some events
  if (k%19!=0) {
    wm = new WorkMsg;
    wm->workerID = (k+20)%numWorkers;
    memset(wm->data, 0, 10*sizeof(int));
    //CkPrintf("At(%d): Team %d(%d) worker %d generated actual work for worker %d\n", ovt, parent->thisIndex, teamID, k, wm->workerID);
    POSE_invoke(work(wm), team, (wm->workerID)/(numWorkers/numTeams), k%50+10);
  }
  if (k%4==0) {
    wm = new WorkMsg;
    wm->workerID = (k+1)%numWorkers;
    memset(wm->data, 0, 10*sizeof(int));
    //CkPrintf("At(%d): Team %d(%d) worker %d generated actual work for worker %d\n", ovt, parent->thisIndex, teamID, k, wm->workerID);
    POSE_invoke(work(wm), team, (wm->workerID)/(numWorkers/numTeams), 100);
  }
  if (k%33==0) {
    wm = new WorkMsg;
    wm->workerID = (k+3)%numWorkers;
    memset(wm->data, 0, 10*sizeof(int));
    //CkPrintf("At(%d): Team %d(%d) worker %d generated actual work for worker %d\n", ovt, parent->thisIndex, teamID, k, wm->workerID);
    POSE_invoke(work(wm), team, (wm->workerID)/(numWorkers/numTeams), k+31);
  }
}
