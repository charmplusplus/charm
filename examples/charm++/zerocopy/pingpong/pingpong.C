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
    CkPrintf("Size\t\tIterations\tRegSend\t\tZCopySend\tRegSendRegRecv\tZCopySendRegRecv\tZCopySendZCopyRecv\n");
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
  double start_time, end_time, reg_time1, reg_time2, zcpy_time1, zcpy_time2, zcpy_time3;
  char *nocopyMsg;

  // other application buffer where the msg is actually expected
  char *otherMsg;

public:
  Ping1()
  {
    nocopyMsg = new char[MAX_PAYLOAD];
    otherMsg = new char[MAX_PAYLOAD];
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
    thisProxy[1].recv_regSend(nocopyMsg, size);
  }

  void freeBuffer(){
    delete [] nocopyMsg;
    if(thisIndex == 0){
      thisProxy[1].freeBuffer();
    }
    else{
      CkExit();
    }
  }

  void recv_regSend(char* msg, int size)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        reg_time1 = 1.0e6*(end_time-start_time)/iterations;
        niter = 0;
        start_time = CkWallTimer();
        thisProxy[1].recv_zcpySend(CkSendBuffer(nocopyMsg), size);
      } else {
        thisProxy[1].recv_regSend(nocopyMsg, size);
      }
    } else {
      thisProxy[0].recv_regSend(nocopyMsg, size);
    }
  }

  void recv_zcpySend(char* msg, int size)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        zcpy_time1 = 1.0e6*(end_time-start_time)/iterations;
        niter=0;
        start_time = CkWallTimer();
        thisProxy[1].recv_regSend_regRecv(nocopyMsg, size);
      } else {
        thisProxy[1].recv_zcpySend(CkSendBuffer(nocopyMsg), size);
      }
    } else {
      thisProxy[0].recv_zcpySend(CkSendBuffer(nocopyMsg), size);
    }
  }

  void recv_regSend_regRecv(char *msg, int size)
  {
    //copy into the user's buffer
    memcpy(otherMsg, msg, size);

    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        reg_time2 = 1.0e6*(end_time-start_time)/iterations;
        niter=0;
        start_time = CkWallTimer();
        thisProxy[1].recv_zcpySend_regRecv(CkSendBuffer(nocopyMsg), size);
      } else {
        thisProxy[1].recv_regSend_regRecv(nocopyMsg, size);
      }
    } else {
      thisProxy[0].recv_regSend_regRecv(nocopyMsg, size);
    }
  }

  void recv_zcpySend_regRecv(char *msg, int size)
  {
    //copy into the user's buffer
    memcpy(otherMsg, msg, size);

    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        zcpy_time2 = 1.0e6*(end_time-start_time)/iterations;
        niter=0;
        start_time = CkWallTimer();
        thisProxy[1].recv_zcpySend_zcpyRecv(CkSendBuffer(nocopyMsg), size);
      } else {
        thisProxy[1].recv_zcpySend_regRecv(CkSendBuffer(nocopyMsg), size);
      }
    } else {
      thisProxy[0].recv_zcpySend_regRecv(CkSendBuffer(nocopyMsg), size);
    }
  }

  void RdmaPost_recv_zcpySend_zcpyRecv(CkRdmaPostStruct *postStruct, int size, CkRdmaPostHandle *handle) {
    postStruct->ptr = otherMsg;
    CkRdmaPost(handle);
  }

  void recv_zcpySend_zcpyRecv(char *msg, int size)
  {
    // message already present in otherMsg because of zcpyRecv
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        zcpy_time3 = 1.0e6*(end_time-start_time)/iterations;
        niter=0;
        if(size < 1 << 24)
          CkPrintf("%d\t\t%d\t\t%lf\t%lf\t%lf\t%lf\t\t%lf\n", size, iterations, reg_time1/2, zcpy_time1/2, reg_time2/2, zcpy_time2/2, zcpy_time3/2);
        else //using different print format for larger numbers for aligned output
          CkPrintf("%d\t%d\t\t%lf\t%lf\t%lf\t%lf\t\t%lf\n", size, iterations, reg_time1/2, zcpy_time1/2, reg_time2/2, zcpy_time2/2, zcpy_time3/2);
        mainProxy.maindone();
      } else {
        thisProxy[1].recv_zcpySend_zcpyRecv(CkSendBuffer(nocopyMsg), size);
      }
    } else {
      thisProxy[0].recv_zcpySend_zcpyRecv(CkSendBuffer(nocopyMsg), size);
    }
  }
};

#include "pingpong.def.h"
