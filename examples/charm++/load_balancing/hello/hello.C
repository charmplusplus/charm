#include "hello.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Hello helloProxy;

class Main : public CBase_Main {
  public:
    Main(CkArgMsg* m) {
      delete m;

      // 2 chares per PE
      int n = 2 * CkNumPes();

      helloProxy = CProxy_Hello::ckNew(n);
      helloProxy.work();
    }

    void done() {
      CkExit();
    }
};

class Hello : public CBase_Hello {
  int pe;

  public:
    Hello() {
      usesAtSync = true;
      pe = CkMyPe();
      CkPrintf("Hello, I'm chare %d on PE %d\n", thisIndex, pe);
    }

    Hello(CkMigrateMessage* m) { }

    void pup(PUP::er &p) {
      p|pe;
    }

    void work() {
      // For chares on latter half of the PEs, introduce artificial load
      // so that they can be migrated to the lower half
      bool heavy = (CkMyPe() >= (CkNumPes() / 2));
      double start_time = CkWallTimer();
      if (heavy) {
        // Busy wait for one second
        while (CkWallTimer() - start_time < 1) { }
      }

      // Informs the runtime system that the chare is ready to migrate
      AtSync();
    }

    void ResumeFromSync() {
      if (CkMyPe() != pe) {
        CkPrintf("I'm chare %d, I moved to PE %d from PE %d\n", thisIndex, CkMyPe(), pe);
      }
      else {
        CkPrintf("I'm chare %d, I'm staying on PE %d\n", thisIndex, pe);
      }

      CkCallback cb(CkReductionTarget(Main, done), mainProxy);
      contribute(cb);
    }
};

#include "hello.def.h"
