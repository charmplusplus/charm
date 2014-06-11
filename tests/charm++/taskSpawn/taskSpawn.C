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
    for (uint64_t i=0; i<tasks; ++i)
      CProxy_worker::ckNew();

    CkCallback endCb(CkIndex_main::results(), thisProxy);
    CkStartQD(endCb);
  }

  void results() {
    //    if (0 == --count) {
      endTime = CkWallTimer();
      CkPrintf("Total execution time: %.2f s\n", endTime - startTime);
      CkExit();
    //    }
  }

};

class worker: public CBase_worker {
public:
  worker(){
    double volatile d = 0.;
    for (uint64_t i=0; i<delay; ++i)
      d += 1. / (2. * i + 1.);
    //    mainProxy.results();
  }
};

#include "taskSpawn.def.h"

