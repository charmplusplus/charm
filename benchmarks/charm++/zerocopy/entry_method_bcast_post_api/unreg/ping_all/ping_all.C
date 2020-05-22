#include <string.h> // for strlen, and strcmp

#include <charm++.h>

#include "ping_all.decl.h"

CProxy_main mainProxy;
int iterations;
size_t minSize, maxSize, smallIter, bigIter;

#define DEBUG(x) //x

class main : public CBase_main
{
  int niter, send_counter, recv_counter;
  char *buffer;
  CProxy_LargeDataNodeGroup ngid;
  double start_time, end_time, reg_time, zcpy_send_time, zcpy_recv_time;
  CkCallback sendDoneCb, recvDoneCb;
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
    send_counter = 0;
    recv_counter = 0;
    mainProxy = thisProxy;
    warmUp = true;
    iterations = smallIter;

    // Allocate a buffer to send
    buffer = new char[maxSize];

    // Create a nodegroup
    ngid = CProxy_LargeDataNodeGroup::ckNew(maxSize);

    // Create a callback method to pass in the Zerocopy Bcast Send API call
    int idx_zerocopySendDone = CkIndex_main::zerocopySendDone();
    sendDoneCb = CkCallback(idx_zerocopySendDone, thisProxy);

    // Create a callback method to pass in the Zerocopy Bcast Recv API call
    int idx_zerocopyRecvDone = CkIndex_main::zerocopyRecvDone();
    recvDoneCb = CkCallback(idx_zerocopyRecvDone, thisProxy);

    CkPrintf("Size (bytes) \t\tIterations\t\tRegular Bcast API (one-way us)\tZero Copy Bcast Send API (one-way us)\tZero Copy Bcast Recv API (one-way us)\n");
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
      ngid.recv_zerocopy(CkSendBuffer(buffer, sendDoneCb, CK_BUFFER_UNREG), size, niter, warmUp, iterations);
    } else {
      ngid.recv(buffer, size, niter, warmUp, iterations);
    }
  }

  void zerocopySendDone() {
    zc_send_done();
  }

  void zerocopyRecvDone() {
    zc_recv_done();
  }

  void zc_send_done() {
    send_counter++;
    if(send_counter == 2) {
      send_counter = 0;
      niter++; // An iteration of the Zerocopy Bcast Send API is complete
      if(niter == iterations) {
        end_time = CkWallTimer();
        zcpy_send_time = 1.0e6*(end_time - start_time)/iterations;
        niter = 0;

        start_time = CkWallTimer();
        ngid.recv_zerocopy_post(CkSendBuffer(buffer, recvDoneCb, CK_BUFFER_UNREG), size, niter, warmUp, iterations);
      } else {
        ngid.recv_zerocopy(CkSendBuffer(buffer, sendDoneCb, CK_BUFFER_UNREG), size, niter, warmUp, iterations);
      }
    }
  }

  void zc_recv_done() {
    recv_counter++;
    if(recv_counter == 2) {
      recv_counter = 0;
      niter++; // An iteration of the Zerocopy Bcast Recv API is complete
      if(niter == iterations) {
        end_time = CkWallTimer();
        zcpy_recv_time = 1.0e6*(end_time - start_time)/iterations;
        niter = 0;

        if(warmUp == false) {
            if(size < 1 << 24)
              CkPrintf("%zu\t\t\t%d\t\t\t%lf\t\t\t%lf\t\t\t\t%lf\n", size, iterations, reg_time, zcpy_send_time, zcpy_recv_time);
            else
              CkPrintf("%zu\t\t%d\t\t\t%lf\t\t\t%lf\t\t\t\t%lf\n", size, iterations, reg_time, zcpy_send_time, zcpy_recv_time);
        }
        size = size << 1;
        if(warmUp)
          done();
        else
          mainProxy.start();
      } else {
        ngid.recv_zerocopy_post(CkSendBuffer(buffer, recvDoneCb, CK_BUFFER_UNREG), size, niter, warmUp, iterations);
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


class LargeDataNodeGroup : public CBase_LargeDataNodeGroup {

  CkCallback regCb, zcSendCb, zcRecvCb;
  char *myBuffer;

public:
  LargeDataNodeGroup(int maxSize) {
    regCb = CkCallback(CkReductionTarget(main, regular_bcast_done), mainProxy);
    zcSendCb = CkCallback(CkReductionTarget(main, zc_send_done), mainProxy);
    zcRecvCb = CkCallback(CkReductionTarget(main, zc_recv_done), mainProxy);

    // allocate a large buffer
    myBuffer = new char[maxSize];
  }

  void recv(char *msg, size_t size, int iter, bool warmUp, int iterations) {
    // manually copy the received message into myBuffer
    memcpy(myBuffer, msg, size);
    contribute(regCb); // Nodegroup reduction to signal completion to the main chare
  }

  void recv_zerocopy(char *msg, size_t size, int iter, bool warmUp, int iterations) {
    // manually copy the received message into myBuffer
    memcpy(myBuffer, msg, size);
    contribute(zcSendCb);
  }

  void recv_zerocopy_post(char *&msg, size_t &size, int iter, bool warmUp, int iterations, CkNcpyBufferPost *postStruct) {
    msg = myBuffer;

    postStruct[0].regMode = CK_BUFFER_UNREG;
  }

  void recv_zerocopy_post(char *msg, size_t size, int iter, bool warmUp, int iterations) {
    contribute(zcRecvCb);
  }
};


#include "ping_all.def.h"

