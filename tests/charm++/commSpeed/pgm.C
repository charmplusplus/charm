#include <unistd.h>
#include <math.h>
#include "pgm.h"
#include "Pgm.def.h"
#include "Worker.h"

CProxy_main mp;

main::main(CkArgMsg *m)
{ 
  mp = thishandle;
  WorkerData *wd;
  if(m->argc<4) {
    CkPrintf("Usage: commSpeed <#remoteMsgs> <#msgSzBytes> <#iterations>\n");
    CkExit();
  }
  numMsgs = atoi(m->argv[1]);
  msgSize = atoi(m->argv[2]);
  numIter = atoi(m->argv[3]);
  CkPrintf(">>> commSpeed test run with %d remote messages of size %d per processor...\n", numMsgs, msgSize);
  numMsgs /= 2;
  if (numMsgs == 0) numMsgs = 1;

  wArray = CProxy_worker::ckNew();
  // create all the workers
  for (int i=0; i<CkNumPes(); i++) {
    wd = new WorkerData;
    wd->numMsgs = numMsgs;
    wd->msgSize = msgSize;
    //CkPrintf("...Creating array element %d on processor %d...\n", i*2, i);
    wArray[i*2].insert(wd, i);
    wd = new WorkerData;
    wd->numMsgs = numMsgs;
    wd->msgSize = msgSize;
    //CkPrintf("...Creating array element %d on processor %d...\n", i*2+1, i);
    wArray[i*2 + 1].insert(wd, i);
  }
  localAvg = remoteAvg = localMax = remoteMax = 0.0;
  localMin = remoteMin = 10000.0;
  initTime = startTime = CkWallTimer();
}

void main::finish(double avgLocal, double avgRemote)
{
  static int remaining = CkNumPes()*2;
  static int iterations=0;
  remaining--;
  localAvg += avgLocal;
  remoteAvg += avgRemote;
  if (remaining == 0) {
    iterations++;
    localAvg = localAvg / (CkNumPes()*2*numMsgs);
    remoteAvg = remoteAvg / (CkNumPes()*2*numMsgs);
    CkPrintf("%d PE Time for Iteration %d= %3.9f\nREMOTE: Avg=%3.9f  LOCAL: Avg=%3.9f\n",
	     CkNumPes(), iterations, CkWallTimer()-startTime, remoteAvg, localAvg);
    if (iterations == numIter) {
      CkPrintf("%d PE Time for all %d iterations: %3.9f\n", CkNumPes(), numIter, CkWallTimer()-initTime);
      CkExit();
    }
    else {
      remaining = CkNumPes()*2;
      localAvg = remoteAvg = 0.0;
      wArray.doStuff();
      startTime = CkWallTimer();
    }
  }
}
