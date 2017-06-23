#include "pingpong.decl.h"

#define BIG_ITER 1000
#define SMALL_ITER 100

#define MAX_PAYLOAD 1 << 27

CProxy_main mainProxy;

#define P1 0
#define P2 1%CkNumPes()

class main : public CBase_main
{
  CProxy_Ping1 arr1;
  int size;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m)
  {
    if(CkNumPes()>2) {
      CkAbort("Run this program on 1 or 2 processors only.\n");
    }
    delete m;
    size = 1024;
    mainProxy = thisProxy;
    CkPrintf("Size (bytes) \t\tIterations\t\tRegular API (one-way us)\tZero Copy API (one-way us)\n");
    arr1 = CProxy_Ping1::ckNew(2);
    CkStartQD(CkCallback(CkIndex_main::maindone(), mainProxy));
  };

  void maindone(void){
    if(size < MAX_PAYLOAD){
      arr1[0].start(size);
      size = size << 1;
    }
    else if(size == MAX_PAYLOAD){
      arr1[0].freeBuffer();
    }
  };
};


class Ping1 : public CBase_Ping1
{
  int size;
  int niter;
  int iterations;
  double start_time, end_time, reg_time, zerocpy_time;
  char *rdmaMsg;

public:
  Ping1()
  {
    rdmaMsg = new char[MAX_PAYLOAD];
    niter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  void start(int size)
  {
    niter = 0;
    if(size >= 1 << 20)
      iterations = SMALL_ITER;
    else
      iterations = BIG_ITER;
    start_time = CkWallTimer();
    thisProxy[1].recv(rdmaMsg, size);
  }

  void freeBuffer(){
    delete [] rdmaMsg;
    if(thisIndex == 0){
      thisProxy[1].freeBuffer();
    }
    else{
      CkExit();
    }
  }

  void recv(char* msg, int size)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        reg_time = 1.0e6*(end_time-start_time)/iterations;
        niter = 0;
        start_time = CkWallTimer();
        thisProxy[1].recv_rdma(rdma(rdmaMsg), size);
      } else {
        thisProxy[1].recv(rdmaMsg, size);
      }
    } else {
      thisProxy[0].recv(rdmaMsg, size);
    }
  }

  void recv_rdma(char* msg, int size)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        zerocpy_time = 1.0e6*(end_time-start_time)/iterations;
        if(size < 1 << 24)
          CkPrintf("%d\t\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zerocpy_time/2);
        else //using different print format for larger numbers for aligned output
          CkPrintf("%d\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zerocpy_time/2);
        niter=0;
        mainProxy.maindone();
      } else {
        thisProxy[1].recv_rdma(rdma(rdmaMsg), size);
      }
    } else {
      thisProxy[0].recv_rdma(rdma(rdmaMsg), size);
    }
  }
};

#include "pingpong.def.h"
