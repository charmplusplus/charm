#include "Worker.h"
#include "Worker.def.h"
#include "pgm.h"

CProxy_worker wArray; 

worker::worker(WorkerData *m)
{
  numMsgs = m->numMsgs;
  msgSize = m->msgSize;
  delete m;
  sent = 0;
  rsum = lsum = rmax = lmax = 0.0;
  rmin = lmin = 10000.0;
  //CkPrintf("Worker %d created on PE %d; numMsgs=%d, msgSize=%d; sending to self.\n", thisIndex, CkMyPe(), msgSize, numMsgs);
  wArray[thisIndex].doStuff();
}

void worker::doStuff()
{
  int i;
  double timer;
  WorkMsg *nm;
  // generate remote events
  timer = CkWallTimer();
  for (i=0; i<numMsgs; i++) {
    nm = new (msgSize/sizeof(int), 0) WorkMsg;  
    memset(nm->data, 0, msgSize);
    wArray[(thisIndex+2)%(CkNumPes()*2)].work(nm);
    //CkPrintf("%d on %d sending %d th remote work to %d\n", thisIndex, CkMyPe(), i, (thisIndex+2)%(CkNumPes()*2));
  }
  timer = CkWallTimer() - timer;
  rsum += timer;

  // generate a local event
  timer = CkWallTimer();
  for (i=0; i<numMsgs; i++) {
    nm = new (msgSize/sizeof(int), 0) WorkMsg;
    memset(nm->data, 0, msgSize);
    if (thisIndex%2 == 0) {
      wArray[thisIndex+1].work(nm);
      //CkPrintf("%d on %d sending %d th local work to %d\n", thisIndex, CkMyPe(), i, thisIndex+1);
    }
    else {
      wArray[thisIndex-1].work(nm);
      //CkPrintf("%d on %d sending %d th local work to %d\n", thisIndex, CkMyPe(), i, thisIndex-1);
    }
  }
  timer = CkWallTimer() - timer;
  lsum += timer;
}  

void worker::work(WorkMsg *m)
{
  //CkPrintf("%d on %d received work\n", thisIndex, CkMyPe());
  sent++;
  if (sent >= numMsgs) {
    mp.finish(lsum, rsum);
    sent = 0;
    rsum = lsum = rmax = lmax = 0.0;
    rmin = lmin = 10000.0;
    return;
  }
}

