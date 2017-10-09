#include "pingpong.decl.h"

#define BIG_ITER 1000
#define SMALL_ITER 100

#define MAX_PAYLOAD 1 << 23

CProxy_main mainProxy;
bool warmUp;

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
    size = 512;
    mainProxy = thisProxy;
    warmUp = true;
    CkPrintf("Size (bytes) \t\tIterations\t\tRegular API (one-way us)\tDirect Get Get (one-way us)\tDirect Put Put (one-way us)\n");
    arr1 = CProxy_Ping1::ckNew(2);
    CkStartQD(CkCallback(CkIndex_main::maindone(), mainProxy));
  };

  void maindone(void){
    if(size < MAX_PAYLOAD) {
      arr1[0].start(size);
      size = size << 1;
    } else if(size == MAX_PAYLOAD) {
      arr1[0].freeBuffer();
    }
  }
};

class Ping1 : public CBase_Ping1
{
  int size;
  int niter;
  int iterations;
  int counter;
  int otherIndex;
  double start_time, end_time, reg_time, zcpy_time1, zcpy_time2;
  CkNcpySource mySrc;
  CkNcpyDestination myDest;
  char *nocopyMsg, *otherMsg;

  public:
  Ping1()
  {
    nocopyMsg = new char[MAX_PAYLOAD];
    otherMsg = new char[MAX_PAYLOAD];
    niter = 0;
    otherIndex = (thisIndex + 1) % 2;
  }

  Ping1(CkMigrateMessage *m) {}

  void start(int size)
  {
    counter = 0;
    this->size = size;
    if(size >= 1 << 20)
      iterations = SMALL_ITER;
    else
      iterations = BIG_ITER;

    start_time = CkWallTimer();
    // send CkNcpySource to 1
    thisProxy[1].recv(nocopyMsg, size);
  }

  void recv(char *msg, int size) {
    //copy into the user's buffer
    memcpy(otherMsg, msg, size);
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        reg_time = 1.0e6*(end_time-start_time)/iterations;
        thisProxy.setupGetGetPingpong(size);
      } else {
        thisProxy[1].recv(nocopyMsg, size);
      }
    } else {
      thisProxy[0].recv(nocopyMsg, size);
    }
  }

  void setupGetGetPingpong(int size) {
    // Source callback and Ncpy object
    CkCallback srcCb = CkCallback(CkCallback::ignore);
    mySrc = CkNcpySource(nocopyMsg, sizeof(char)*size, srcCb);

    // Destination callback and Ncpy object
    CkCallback destCb = CkCallback(CkIndex_Ping1::callbackGetGetPingpong(NULL), thisProxy[thisIndex]);
    myDest = CkNcpyDestination(otherMsg, sizeof(char)*size, destCb);

    thisProxy[0].beginGetGetPingpong();
  }

  void beginGetGetPingpong() {
    counter++;
    if(counter == 2) {
      niter=0;
      start_time = CkWallTimer();
      thisProxy[1].recvNcpySrcInfo(mySrc);
    }
  }

  void recvNcpySrcInfo(CkNcpySource otherSrc) {
    myDest.rget(otherSrc);
  }

  void callbackGetGetPingpong(CkDataMsg *m) {
    if(thisIndex == 0) {
      // iteration completed
      niter++;
      if(niter == iterations) {
        end_time = CkWallTimer();
        zcpy_time1 = 1.0e6*(end_time - start_time)/iterations;
        niter = 0;
        thisProxy.endGetGetPingpong();
      } else {
        thisProxy[1].recvNcpySrcInfo(mySrc);
      }
    } else {
      thisProxy[0].recvNcpySrcInfo(mySrc);
    }
  }

  void endGetGetPingpong() {
    counter = 0;
    mySrc.releaseResource();
    myDest.releaseResource();
    thisProxy[0].doneGetGetPingpong();
  }

  void doneGetGetPingpong() {
    counter++;
    if(counter == 2) {
      counter = 0;
      thisProxy.setupPutPutPingpong(size);
    }
  }

  void setupPutPutPingpong(int size) {

    // Source callback and Ncpy object
    CkCallback srcCb = CkCallback(CkCallback::ignore);
    mySrc = CkNcpySource(nocopyMsg, sizeof(char)*size, srcCb);

    // Destination callback and Ncpy object
    CkCallback destCb = CkCallback(CkIndex_Ping1::callbackPutPutPingpong(NULL), thisProxy[thisIndex]);
    myDest = CkNcpyDestination(otherMsg, sizeof(char)*size, destCb);

    thisProxy[0].beginPutPutPingpong();
  }

  void beginPutPutPingpong() {
    counter++;
    if(counter == 2) {
      counter = 0;
      niter=0;
      start_time = CkWallTimer();
      thisProxy[1].askNcpyDestInfo();
    }
  }

  void askNcpyDestInfo() {
    thisProxy[otherIndex].recvNcpyDestInfo(myDest);
  }

  void recvNcpyDestInfo(CkNcpyDestination otherDest) {
    mySrc.rput(otherDest);
  }

  void callbackPutPutPingpong(CkDataMsg *m) {
    if(thisIndex == 0) {
      // iteration completed
      niter++;
      if(niter == iterations) {
        end_time = CkWallTimer();
        zcpy_time2 = 1.0e6*(end_time - start_time)/iterations;
        niter = 0;
        if(warmUp) {
          warmUp = false;
        } else {
          if(size < 1 << 24)
            CkPrintf("%d\t\t\t%d\t\t\t%lf\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zcpy_time1/2, zcpy_time2/2);
          else //using different print format for larger numbers for aligned output
            CkPrintf("%d\t\t%d\t\t\t%lf\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zcpy_time1/2, zcpy_time2/2);
        }
        thisProxy.endPutPutPingpong();
      } else {
        thisProxy[1].askNcpyDestInfo();
      }
    } else {
      thisProxy[0].askNcpyDestInfo();
    }
  }

  void endPutPutPingpong() {
    mySrc.releaseResource();
    myDest.releaseResource();
    thisProxy[0].donePutPutPingpong();
  }

  void donePutPutPingpong() {
    counter++;
    if(counter == 2) {
      counter = 0;
      mainProxy.maindone();
    }
  }

  void freeBuffer(){
    delete [] nocopyMsg;
    delete [] otherMsg;
    if(thisIndex == 0){
      thisProxy[1].freeBuffer();
    }
    else{
      CkExit();
    }
  }
};

#include "pingpong.def.h"
