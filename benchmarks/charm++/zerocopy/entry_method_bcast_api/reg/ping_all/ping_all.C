#include <string.h> // for strlen, and strcmp
#include <charm++.h>

#include "ping_all.decl.h"

CProxy_main mainProxy;
int iterations;
size_t minSize, maxSize, smallIter, bigIter;

#define DEBUG(x) //x

class main : public CBase_main
{
  int niter, counter;
  char *buffer;
  CProxy_LargeDataNodeGroup ngid;
  double start_time, end_time, reg_time, zcpy_time;
  CkCallback cb;
  size_t size;
  bool warmUp;

public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    if(m->argc == 5) {
      minSize = atoi(m->argv[1])/2; // Start with a smaller size to run a warm up phase
      maxSize = atoi(m->argv[2]);
      smallIter = atoi(m->argv[3]);
      bigIter = atoi(m->argv[4]);
    } else if(m->argc == 1) {
      // use defaults
      minSize = 512; // Start with a smaller size to run a warm up phase before starting message size at 1024 bytes
      maxSize = 1 << 13;
      smallIter = 10;
      bigIter = 100;
    } else {
      CkPrintf("Usage: ./ping_all <min size> <max size> <small message iter> <big message iter>\n");
      CkExit(1);
    }
    delete m;
    // Initialize
    size = minSize;
    niter = 0;
    counter = 0;
    mainProxy = thisProxy;
    warmUp = true;
    iterations = smallIter;

    // Allocate a buffer to send
    buffer = new char[maxSize];

    // Create a nodegroup
    ngid = CProxy_LargeDataNodeGroup::ckNew();

    // Create a callback method to pass in the Zerocopy Bcast API call
    int idx_zerocopySent = CkIndex_main::zerocopySent();
    cb = CkCallback(idx_zerocopySent, thisProxy);

    CkPrintf("Size (bytes) \t\tIterations\t\tRegular Bcast API (one-way us)\tZero Copy Bcast Send API (one-way us)\t\n");
    CkStartQD(CkCallback(CkIndex_main::start(), mainProxy));
  }

  void start() {
    if(size < minSize) {
      // warmUp phase
      start_time = CkWallTimer();
      ngid.recv(buffer, size, niter, warmUp, iterations);
    } else if(size <= maxSize) {
      // regular experiment phase
      start_time = CkWallTimer();
      ngid.recv(buffer, size, niter, warmUp, iterations);
    } else {
      // completion phase
      done();
    }
  }

  // Invoked on main after a reduction by all the nodegroup elements
  void regular_bcast_done() {
    niter++; // An iteration of the Regular Bcast API is complete
    if(niter == iterations) {
      end_time = CkWallTimer();
      reg_time = 1.0e6*(end_time - start_time)/iterations;
      niter = 0;
      start_time = CkWallTimer();
      ngid.recv_zerocopy(CkSendBuffer(buffer, cb), size, niter, warmUp, iterations);
    } else {
      ngid.recv(buffer, size, niter, warmUp, iterations);
    }
  }

  void zerocopySent() {
    zc_bcast_done();
  }

  void zc_bcast_done() {
    counter++;
    if(counter == 2) {
      counter = 0;
      niter++; // An iteration of the Zerocopy Bcast API is complete
      if(niter == iterations) {
        end_time = CkWallTimer();
        zcpy_time = 1.0e6*(end_time - start_time)/iterations;
        niter = 0;
        if(warmUp == false) {
            if(size < 1 << 24)
              CkPrintf("%zu\t\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time, zcpy_time);
            else
              CkPrintf("%zu\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time, zcpy_time);
        }
        size = size << 1;
        if(warmUp)
          done();
        else
          mainProxy.start();
      } else {
        ngid.recv_zerocopy(CkSendBuffer(buffer, cb), size, niter, warmUp, iterations);
      }
    }
  }

  void done() {
    if(warmUp) {
      // warmUp phase complete
      warmUp = false;
      mainProxy.start();
    } else {
      // experiment complete
      CkExit();
    }
  }
};


class LargeDataNodeGroup : public CBase_LargeDataNodeGroup
{
  CkCallback regCb, zcCb;

public:
  LargeDataNodeGroup() {
    regCb = CkCallback(CkReductionTarget(main, regular_bcast_done), mainProxy);
    zcCb = CkCallback(CkReductionTarget(main, zc_bcast_done), mainProxy);
  }

  void recv(char *msg, size_t size, int iter, bool warmUp, int iterations) {
    contribute(regCb); // Nodegroup reduction to signal completion to the main chare
  }

  void recv_zerocopy(char *msg, size_t size, int iter, bool warmUp, int iterations) {
    contribute(zcCb);
  }
};


#include "ping_all.def.h"
