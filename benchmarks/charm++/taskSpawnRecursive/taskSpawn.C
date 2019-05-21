#include "taskSpawn.decl.h"

CProxy_main mainProxy;
int delay;

class main: public CBase_main {

  int count;
  int tasks;
  double startTime, endTime;

public:

  main(CkArgMsg*m){
    startTime = CkWallTimer();
    mainProxy = thishandle;
    tasks = atoi(m->argv[1]);
    delay = atoi(m->argv[2]);

    count = tasks;
    CProxy_worker::ckNew(0, tasks - 1);

    CkCallback endCb(CkIndex_main::results(), thisProxy);
    CkStartQD(endCb);
  }

  void results() {
    endTime = CkWallTimer();
    CkPrintf("Total execution time: %.2f s\n", endTime - startTime);
    CkExit();
  }

};

class worker: public CBase_worker {
public:
  worker(int lowerIndex, int upperIndex){
    while (lowerIndex != upperIndex) {
      int midIndex = (lowerIndex + upperIndex + 1) / 2;
      CProxy_worker::ckNew(midIndex, upperIndex);
      upperIndex = midIndex - 1;
    }

    double volatile d = 0.;
    for (uint64_t i=0; i<delay; ++i)
      d += 1. / (2. * i + 1.);
  }
};

#include "taskSpawn.def.h"

