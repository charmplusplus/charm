#include "iterative.decl.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Test test_proxy;
/* readonly */ int n_iters;
/* readonly */ int data_size;

class Main : public CBase_Main {
  double start_time;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    n_iters = 1;
    data_size = 128;

    // Check if there are 2 PEs
    if (CkNumPes() != 2) {
      CkAbort("Should be run with 2 PEs");
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "i:s:")) != -1) {
      switch (c) {
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 's':
          data_size = atoi(optarg);
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("[Load balancing itertions test]\n"
        "Iters: %d, Data size: %d bytes\n", n_iters, data_size);

    // Create chares
    test_proxy = CProxy_Test::ckNew(CkNumPes());

    // Begin testing
    thisProxy.test();
  }

  void test() {
    start_time = CkWallTimer();

    CkPrintf("Testing chare array... ");
    for (int i = 0; i < n_iters; i++) {
      // XXX: Hangs after 1st LB with CkCallbackResumeThread.
      //      With CkWaitQD, LB is performed only once.
      test_proxy[0].send(CkCallbackResumeThread());
      //CkWaitQD();
    }
    CkPrintf("PASS\n");

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class Test : public CBase_Test {
  int pe;
  CkCallback cb;

public:
  Test() {
    usesAtSync = true;
  }

  Test(CkMigrateMessage* m) {}

  void pup(PUP::er& p) {
    p|pe;
    p|cb;
  }

  void send(CkCallback cb) {
    char data[data_size];
    thisProxy[1].recv(cb, data_size, data);
    pe = CkMyPe();
    AtSync();
  }

  void recv(CkCallback cb, int size, char* data) {
    this->cb = cb;
    pe = CkMyPe();
    AtSync();
  }

  void ResumeFromSync() {
    if (thisIndex == 1) cb.send();
  }
};

#include "iterative.def.h"
