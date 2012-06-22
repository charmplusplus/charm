#include <string.h> // for strlen, and strcmp
#include <charm++.h>

#define NITER 1000
#define PAYLOAD 100

#include "ping.decl.h"
class PingMsg : public CMessage_PingMsg
{
  public:
    char *x;

};

CProxy_main mainProxy;
int iterations;
int payload;
int PEsPerNode;
int CharesPerPE;

class main : public CBase_main
{
  int niter;
  CProxy_Ping1 arr1;
  double start_time;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
      if(m->argc < 5)
          CkPrintf("Usage: payload PEs CharesPerPE iteration\n");

      niter = 0;
      iterations=NITER;
      payload=PAYLOAD;
      if(m->argc>1)
          payload=atoi(m->argv[1]);
      if(m->argc>2)
          PEsPerNode = atoi(m->argv[2]);
      if(PEsPerNode > CkMyNodeSize())
          PEsPerNode = CkMyNodeSize();
      if(m->argc>3)
          CharesPerPE = atoi(m->argv[3]);
      if(m->argc>4)
          iterations=atoi(m->argv[4]);
  
      mainProxy = thishandle;
      arr1 = CProxy_Ping1::ckNew(2* PEsPerNode * CharesPerPE );
      start_time = CkWallTimer();
      for(int i=0; i<PEsPerNode * CharesPerPE; i++)
          arr1[i].start();
      delete m;
  };

  void maindone(void)
  {
      niter++;
      if(niter == iterations)
      {
          double pingTimer = CkWallTimer() - start_time;
          CkPrintf("Ping time for %d messages(%d Chares * %d PEs per node) of %d Bytes cost %f ms\n", 
              PEsPerNode * CharesPerPE, CharesPerPE, PEsPerNode, payload, 1000*pingTimer/iterations);
          CkExit();
      }else
      {
          for(int i=0; i<PEsPerNode * CharesPerPE; i++)
              arr1[i].start();
      }
  };
};


class Ping1 : public CBase_Ping1
{
  CProxy_Ping1 *pp;
  int recvCnt;
  double start_time, end_time;
public:
  Ping1()
  {
    pp = new CProxy_Ping1(thisArrayID);
    recvCnt = 0;
  }
  Ping1(CkMigrateMessage *m) {}
  void start(void)
  {
      int  receiver = PEsPerNode * CharesPerPE;
      (*pp)[receiver].recv(new (payload) PingMsg);
  }
  void recv(PingMsg *msg)
  {
      recvCnt++;
      if(recvCnt == PEsPerNode * CharesPerPE)
      {
          mainProxy.maindone();
          recvCnt = 0;
      }
  }
};

#include "ping.def.h"
