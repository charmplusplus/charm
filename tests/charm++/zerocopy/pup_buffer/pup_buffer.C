#include "pup_buffer.decl.h"
#define SIZE 1000000
#define TOTAL_ITER 10

CProxy_arr arrProxy;
CProxy_main mainProxy;
int totalElems;

class main : public CBase_main {
  public:
    main(CkArgMsg *m) {
      delete m;

      mainProxy = thisProxy;

      totalElems = 1 * CkNumPes();
      arrProxy = CProxy_arr::ckNew(totalElems);
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
      buffer = (int *)malloc(sizeof(int) * SIZE);
      iteration = 0;

      usesAtSync = true;

      for(int i=0; i < SIZE; i++)
        buffer[i] = thisIndex;
      cb = CkCallback(CkReductionTarget(main, done), mainProxy);
    }

    arr(CkMigrateMessage *msg) {
      delete msg;
    }

    ~arr() {}

    static void *buff_allocate(size_t n) {
      return new int[n];
    }

    static void buff_deallocate(void *ptr) {
      delete [] (int *)(ptr);
    }

    void pup(PUP::er &p) {
      p|iteration;
      p|cb;

      if(iteration % 2 == 0) // Test pup_buffer custom
        p.pup_buffer(buffer, SIZE, buff_allocate, buff_deallocate);
      else // Test pup_buffer default
        p.pup_buffer(buffer, SIZE);

    }

    void verify() {
      for(int i=0; i < SIZE; i++) {
        CkAssert(buffer[i] == thisIndex);
      }
      run();
    }

    void run() {
      iteration++;
      if(thisIndex == 0)
        CmiPrintf("Iteration %d completed\n", iteration);
      if(iteration < TOTAL_ITER) {
        if(iteration % 5 == 0)
          AtSync();
        else
          thisProxy[(thisIndex + 1) % totalElems].verify();
      } else {
        free(buffer);
        contribute(cb);
      }
    }

    void ResumeFromSync() {
      run();
    }
};

#include "pup_buffer.def.h"
