#include <string.h> // for strlen, and strcmp
#include <charm++.h>
#include <TopoManager.h>
#define NITER 1000
#define PAYLOAD 1000
#define  WARM_UP 100

#define  START_TRACE_ITER       300
#define  END_TRACE_ITER         320

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
  CProxy_TraceControl _traceControl;
  double start_time;
  int nodeIndex;
  int totalPayload;
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
          totalPayload=atoi(m->argv[1]);
      if(m->argc>2)
          PEsPerNode = atoi(m->argv[2]);
      if(m->argc>3)
          CharesPerPE = atoi(m->argv[3]);
      if(m->argc>4)
          iterations=atoi(m->argv[4]);
 
      payload = totalPayload/PEsPerNode/CharesPerPE;
      mainProxy = thishandle;
      arr1 = CProxy_Ping1::ckNew(CkNumNodes()* PEsPerNode * CharesPerPE );
      start_time = CkWallTimer();
      nodeIndex = 1;
      int x,y,z,t;
      TopoManager tmgr;
      for(int i=0; i<CmiNumPes(); i+=CmiMyNodeSize())
      {
          tmgr.rankToCoordinates(i, x,y, z, t);
          CkPrintf(" %d  [%d:%d:%d:%d]\n", i, x, y, z, t);
      }
      CkPrintf("NodeIndex Chares       Workers        NoOfMsgs        Bytes           Total           Time(us)\n");
      _traceControl = CProxy_TraceControl::ckNew();
      for(int i=0; i<PEsPerNode * CharesPerPE; i++)
          arr1[i].start(nodeIndex);
      delete m;
  };

  void maindone(void)
  {
      niter++;
      if(niter == START_TRACE_ITER)
          _traceControl.startTrace();
      if(niter == END_TRACE_ITER)
          _traceControl.endTrace();

      if(niter == iterations)
      {
          double pingTimer = CkWallTimer() - start_time;
          CkPrintf("Pingping %d\t\t %d  \t\t%d  \t\t%d  \t\t%d \t\t%.1f\n",
              nodeIndex, CharesPerPE, PEsPerNode, PEsPerNode * CharesPerPE, payload, 1000*1000*pingTimer/(iterations-WARM_UP));
          if(nodeIndex == CkNumNodes() -1)
              CkExit();
          else
          {
              niter = 0;
              for(int i=0; i<PEsPerNode * CharesPerPE; i++)
                  arr1[i].start(nodeIndex);
          }
          nodeIndex++;
      }else 
      {
          if(niter == WARM_UP)
              start_time = CkWallTimer();
          for(int i=0; i<PEsPerNode * CharesPerPE; i++)
              arr1[i].start(nodeIndex);
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
  void start(int node)
  {
      int  receiver = PEsPerNode * CharesPerPE * node;
      (*pp)[receiver].recv(new (payload) PingMsg);
  }
  void recv(PingMsg *msg)
  {
      delete msg;
      recvCnt++;
      if(recvCnt == PEsPerNode * CharesPerPE)
      {
          mainProxy.maindone();
          recvCnt = 0;
      }
  }
};

class TraceControl : public Group 
{
public:
    TraceControl() { }

    void startTrace() { traceBegin(); }
    
    void endTrace() { traceEnd(); }
};


#include "ping.def.h"
