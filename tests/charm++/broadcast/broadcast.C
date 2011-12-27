#include <string.h> // for strlen, and strcmp
#include <charm++.h>

#define NITER 1000
#define PAYLOAD 100
#define  WARM_UP 100
#define SIMPLE_ACK 0

#include "broadcast.decl.h"
class PingMsg : public CMessage_PingMsg
{
  public:
      int source;
      char *x;

};
class DUMMY : public CMessage_DUMMY {
};

CProxy_main mainProxy;
int iterations;
int payload;

class main : public CBase_main
{
  int phase;
  CProxy_Ping1 arr1;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {

    iterations=NITER;
    payload=PAYLOAD;
    if(m->argc>1)
      payload=atoi(m->argv[1]);
    if(m->argc>2)
      iterations=atoi(m->argv[2]);
    if(m->argc>3)
      CkPrintf("Usage: pgm +pN [payload] [iterations]\n Where N [1-2], payload (default %d) is integer >0 iterations (default %d) is integer >0 ", PAYLOAD, NITER);
    CkPrintf("broadcast with payload: %d iterations: %d\n", payload,iterations);
    mainProxy = thishandle;
    phase = 0;
    arr1 = CProxy_Ping1::ckNew(CkNumPes());
    phase=0;
    mainProxy.maindone();
    delete m;
  }

  void maindone(void)
  {
    switch(phase++) {
      case 0:
	arr1[0].start();
    break;
      default:
    CkExit();
    }
  }
};


class Ping1 : public CBase_Ping1
{
  CProxy_Ping1 *pp;
  int niter, ackCounter;
  double start_time, end_time;
public:
  Ping1()
  {
    pp = new CProxy_Ping1(thisArrayID);
    niter = 0;
  }
  Ping1(CkMigrateMessage *m) {}
  void start(void)
  {
      int i;
      ackCounter = 0;
      for(i=1; i<CkNumPes(); i++)
      {
          (*pp)[i].recv(new (payload) PingMsg);
      }
    start_time = CkWallTimer();
  }
#if SIMPLE_ACK
  void back(PingMsg *msg) {}
  void ack(DUMMY *msg) {
#else
  void ack(DUMMY *msg) {}
  void back(PingMsg *msg) {
#endif
      int i;
      delete msg;
      ackCounter++;
      if(ackCounter == CkNumPes()-1)
      {
        niter++;
        if(niter==iterations+WARM_UP) {
            end_time = CkWallTimer();
            CkPrintf("Time for 1D Arrays broadcast and reduction is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);

            //mainProxy.maindone();
            niter=0;
            start_time = CkWallTimer();
            mainProxy.maindone();
        }else
        {
            if(niter==WARM_UP)
                start_time = CkWallTimer();

            for(i=1; i<CkNumPes(); i++)
            {
                (*pp)[i].recv(new (payload) PingMsg);
            }
            ackCounter =0;
        }
    }
  }
  void recv(PingMsg *msg)
  {
#if SIMPLE_ACK
      delete msg;
      (*pp)[0].back(new DUMMY);
#else
      (*pp)[0].back(msg);
#endif
  }

};

#include "broadcast.def.h"
