#include "vecadd.decl.h"
#include "vecadd.h"
#include <unistd.h>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Process processProxy;
/* readonly */ uint64_t n;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    n = 128 * 1024 * 1024; // 128 M doubles by default

    int c;
    while ((c = getopt(m->argc, m->argv, "n:")) != -1) {
      switch (c) {
        case 'n':
          n = atoi(optarg);
          break;
        default:
          CkExit();
      }
    }

    // Create nodegroup and run
    processProxy = CProxy_Process::ckNew();
    processProxy.run();
  };

  void done() {
    CkPrintf("All done\n");

    CkExit();
  };
};

class Process : public CBase_Process {
public:
  Process() { }

  void run() {
    // OpenACC vector addition
    vecadd(n);

    // Reduce to Main to end the program
    CkCallback cb(CkReductionTarget(Main, done), mainProxy);
    contribute(cb);
  }
};

#include "vecadd.def.h"
