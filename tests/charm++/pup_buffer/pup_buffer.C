#include "pup_buffer.decl.h"
#define SIZE 1000
#define TOTAL_ITER 10

CProxy_arr arrProxy;
CProxy_main mainProxy;

class main : public CBase_main {
  public:
    main(CkArgMsg *m) {
      delete m;

      mainProxy = thisProxy;

      arrProxy = CProxy_arr::ckNew(10 * CkNumPes());
      arrProxy.run();
    }

    void done() {
      CmiPrintf("[%d][%d][%d] Completed testing, Exiting\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
      CkExit();
    }
};

class arr : public CBase_arr {
  int *buffer;
  CkCallback cb;
  int iteration;
  public:
    arr() {
      buffer = new int[SIZE];
      iteration = 0;

      usesAtSync = true;

      for(int i=0; i < SIZE; i++) buffer[i] = thisIndex;
      cb = CkCallback(CkReductionTarget(main, done), mainProxy);
    }

    arr(CkMigrateMessage *msg) {
      delete msg;
    }

    static void *buff_allocate(size_t n) {
      return new int[n];
    }

    static void buff_deallocate(void *ptr) {
      free(ptr);
    }

    void pup(PUP::er &p) {
      p|iteration;
      p|cb;

      if(iteration % 2 == 0) // Test pup_buffer custom
        p.pup_buffer(buffer, SIZE, buff_allocate, buff_deallocate);
      else // Test pup_buffer default
        p.pup_buffer(buffer, SIZE);

      if(p.isUnpacking())
        for(int i=0; i < SIZE; i++)
          CkAssert(buffer[i] == thisIndex);
    }

    void run() {
      iteration++;
      if(thisIndex == 0)
        CmiPrintf("Iteration %d completed\n", iteration);
      if(iteration < TOTAL_ITER) {
        AtSync();
      } else {
        contribute(cb);
      }
    }

    void ResumeFromSync() {
      run();
    }
};

#include "pup_buffer.def.h"
