#include "pingpong.decl.h"

CProxy_main mainProxy;
int minSize, maxSize, smallIter, bigIter;

#define P1 0
#define P2 1%CkNumPes()

class main : public CBase_main
{
  CProxy_Ping1 arr1;
  int size;
  bool warmUp;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m)
  {
    if(CkNumPes()>2) {
      CkAbort("Run this program on 1 or 2 processors only.\n");
    }
    if(m->argc == 5) {
      minSize = atoi(m->argv[1])/2; // Start with a smaller size to run a warm up phase
      maxSize = atoi(m->argv[2]);
      smallIter = atoi(m->argv[3]);
      bigIter = atoi(m->argv[4]);
    } else if(m->argc == 1) {
      // use defaults
      minSize = 512; // Start with a smaller size to run a warm up phase before starting message size at 1024 bytes
      maxSize = 1 << 25;
      smallIter = 1000;
      bigIter = 100;
    } else {
      CkPrintf("Usage: ./pingpong <min size> <max size> <small message iter> <big message iter>\n");
      CkExit(1);
    }
    delete m;
    size = minSize;
    mainProxy = thisProxy;
    warmUp = true;
    CkPrintf("Size (bytes) \t\tIterations\t\tRegular API (one-way us)\tZero Copy API (one-way us)\n");
    arr1 = CProxy_Ping1::ckNew(2);
    CkStartQD(CkCallback(CkIndex_main::maindone(), mainProxy));
  };

  void maindone(void){
    if(size <= maxSize) {
      arr1[0].start(size, warmUp);
      warmUp = false;
      size = size << 1;
    } else {
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
  char *nocopyMsg;

public:
  Ping1()
  {
    nocopyMsg = new char[maxSize];
    niter = 0;
  }
  Ping1(CkMigrateMessage *m) {}

  void start(int size, bool warmUp)
  {
    niter = 0;
    if(size >= 1 << 20)
      iterations = bigIter;
    else
      iterations = smallIter;
    start_time = CkWallTimer();
    thisProxy[1].recv(nocopyMsg, size, warmUp);
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

  void recv(char* msg, int size, bool warmUp)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        reg_time = 1.0e6*(end_time-start_time)/iterations;
        niter = 0;
        start_time = CkWallTimer();
        thisProxy[1].recv_zerocopy(CkSendBuffer(nocopyMsg), size, warmUp);
      } else {
        thisProxy[1].recv(nocopyMsg, size, warmUp);
      }
    } else {
      thisProxy[0].recv(nocopyMsg, size, warmUp);
    }
  }

  void recv_zerocopy(char* msg, int size, bool warmUp)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        zerocpy_time = 1.0e6*(end_time-start_time)/iterations;
        if(warmUp == false) {
          if(size < 1 << 24)
            CkPrintf("%d\t\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zerocpy_time/2);
          else //using different print format for larger numbers for aligned output
            CkPrintf("%d\t\t%d\t\t\t%lf\t\t\t%lf\n", size, iterations, reg_time/2, zerocpy_time/2);
        }
        niter=0;
        mainProxy.maindone();
      } else {
        thisProxy[1].recv_zerocopy(CkSendBuffer(nocopyMsg), size, warmUp);
      }
    } else {
      thisProxy[0].recv_zerocopy(CkSendBuffer(nocopyMsg), size, warmUp);
    }
  }
};

#include "pingpong.def.h"
