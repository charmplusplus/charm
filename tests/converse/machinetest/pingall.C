/****************************************************************
           Benchmark tests the performance of PINGPONG with 
           oneway and two way traffic. It reports both completion 
           time and COU overhead. The CPU overhead is computed from 
           the idle time, and hence can be inaccurate on some machine layers.

   - Sameer Kumar (03/08/05)
*******************************************************************/


#include <stdlib.h>
#include <converse.h>
#define ENABLE_TIMER 0

//Iterations for each message size
enum {nCycles = 200};

//Max message size
enum { maxMsgSize = 1 << 18 };

CpvDeclare(int,msgSize);
CpvDeclare(int,cycleNum);
CpvDeclare(int,sizeNum);
CpvDeclare(int, ackCount);
CpvDeclare(int, twoway);
CpvDeclare(int,exitHandler);
CpvDeclare(int,node0Handler);
CpvDeclare(int,node1Handler);
CpvDeclare(int,ackHandler);
CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);

CpvDeclare(double, IdleStartTime);
CpvDeclare(double, IdleTime);

//Registering idle handlers
void ApplIdleStart(void *, double start)
{
    CpvAccess(IdleStartTime)= start; //CmiWallTimer();
    return;
}

void ApplIdleEnd(void *, double cur)
{
  if(CpvAccess(IdleStartTime) < 0)
      return;
  
  CpvAccess(IdleTime) += cur /*CmiWallTimer()*/-CpvAccess(IdleStartTime);
  CpvAccess(IdleStartTime)=-1;
  return;
}

//Start ping pong

void startPingpong()
{
    CpvAccess(cycleNum) = 0;
    CpvAccess(msgSize) = (CpvAccess(msgSize)-CmiMsgHeaderSizeBytes)*2 + 
        CmiMsgHeaderSizeBytes;

    char *msg = (char *)CmiAlloc(CpvAccess(msgSize));
    *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
  
    CmiSetHandler(msg,CpvAccess(node0Handler));
    CmiSyncSendAndFree(CmiMyPe(), CpvAccess(msgSize), msg);
    
    CpvAccess(startTime) = CmiWallTimer();
    CpvAccess(IdleTime) = 0.0;
}

void pingpongFinished(char *msg)
{
    CmiFree(msg);
    double cycle_time = 
        (1e6*(CpvAccess(endTime)-CpvAccess(startTime)))/(2.*nCycles);
    double compute_time = cycle_time - 
        (1e6*(CpvAccess(IdleTime)))/(2.*nCycles);

#if ENABLE_TIMER
    CmiPrintf("[%d] %d \t %5.3lfus \t %5.3lfus\n", CmiMyPe(),
              CpvAccess(msgSize) - CmiMsgHeaderSizeBytes, 
              cycle_time, compute_time);
#endif
    
    CpvAccess(sizeNum)++;
    
    if (CpvAccess(msgSize) < maxMsgSize)
        startPingpong();
    else {
      void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(sendmsg,CpvAccess(ackHandler));
      CmiSyncSendAndFree(0, CmiMsgHeaderSizeBytes, sendmsg);
    }
}

CmiHandler ackHandlerFunc(char *msg)
{
    CmiFree(msg);
    CpvAccess(ackCount)++;
    int max = CpvAccess(twoway) ? CmiNumPes() : CmiNumPes()/2;
    if(CpvAccess(ackCount) == max) {
      void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(sendmsg,CpvAccess(exitHandler));
      CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
    }
    return 0;
}

CmiHandler exitHandlerFunc(char *msg)
{
    CmiFree(msg);
    CsdExitScheduler();
    return 0;
}

CmiHandler node0HandlerFunc(char *msg)
{
    CpvAccess(cycleNum)++;
    
    if (CpvAccess(cycleNum) == nCycles) {
        CpvAccess(endTime) = CmiWallTimer();
        pingpongFinished(msg);
    }
    else {
        CmiSetHandler(msg,CpvAccess(node1Handler));
        *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
        
        int dest = CmiNumPes() - CmiMyPe() - 1;
        CmiSyncSendAndFree(dest,CpvAccess(msgSize),msg);
    }
    
    return 0;
}

CmiHandler node1HandlerFunc(char *msg)
{
    int msgSize = *((int *)(msg+CmiMsgHeaderSizeBytes));
    CmiSetHandler(msg,CpvAccess(node0Handler));
    
    int dest = CmiNumPes() - CmiMyPe() - 1;
    CmiSyncSendAndFree(dest,msgSize,msg);
    return 0;
}

CmiStartFn mymain(int argc, char** argv)
{
    if(CmiMyRank() == CmiMyNodeSize()) return 0;

    CpvInitialize(int,msgSize);
    CpvInitialize(int,cycleNum);
    CpvInitialize(int,sizeNum);
    CpvAccess(sizeNum) = 1;
    CpvAccess(msgSize)= CmiMsgHeaderSizeBytes + 8;
    
    CpvInitialize(int,exitHandler);
    CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
    CpvInitialize(int,node0Handler);
    CpvAccess(node0Handler) = CmiRegisterHandler((CmiHandler) node0HandlerFunc);
    CpvInitialize(int,node1Handler);
    CpvAccess(node1Handler) = CmiRegisterHandler((CmiHandler) node1HandlerFunc);
    CpvInitialize(int,ackHandler);
    CpvAccess(ackHandler) = CmiRegisterHandler((CmiHandler) ackHandlerFunc);
    
    CpvInitialize(double,startTime);
    CpvInitialize(double,endTime);
    
    CpvInitialize(double, IdleStartTime);
    CpvInitialize(double, IdleTime);

    CpvInitialize(int,ackCount);
    CpvAccess(ackCount) = 0;

    CpvInitialize(int,twoway);
    CpvAccess(twoway) = 0;

    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, ApplIdleStart, NULL);
    CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE, ApplIdleEnd, NULL);
    
    if(argc > 1)
        CpvAccess(twoway) = atoi(argv[1]);

    if(CmiMyPe() == 0) {
      if(!CpvAccess(twoway))
        CmiPrintf("Starting Pingpong with oneway traffic \n");
      else
        CmiPrintf("Starting Pingpong with twoway traffic\n");
    }

    if ((CmiMyPe() < CmiNumPes()/2) || CpvAccess(twoway))
      startPingpong();

    return 0;
}

int main(int argc,char *argv[])
{
    ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
    return 0;
}
