#include <stdlib.h>
#include <converse.h>

enum {nCycles = 1 << 8 };
enum { maxMsgSize = 1 << 13 };

CpvDeclare(int,msgSize);
CpvDeclare(int,cycleNum);
CpvDeclare(int,sizeNum);
CpvDeclare(int,exitHandler);
CpvDeclare(int,node0Handler);
CpvDeclare(int,node1Handler);
CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);
CpvStaticDeclare(double,startCTime);
CpvStaticDeclare(double,endCTime);

void startRing()
{
  CpvAccess(cycleNum) = 0;
  CpvAccess(msgSize) = 100 * CpvAccess(sizeNum) + CmiMsgHeaderSizeBytes;
  //CmiPrintf("PE %d startRing allocating %d bytes, header=%d bytes\n",
	    //CmiMyPe(),CpvAccess(msgSize),CmiMsgHeaderSizeBytes);
  CmiPrintf(
  "       cycles       bytes         Total(ms)     One-way (us/msg)\n"
  );
  char *msg = (char *)CmiAlloc(CpvAccess(msgSize));
  *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
  CmiPrintf("PE %d startRing starting clock\n",CmiMyPe());
  CpvAccess(startTime) = CmiWallTimer();
  CpvAccess(startCTime) = CmiTimer();
  CmiSetHandler(msg,CpvAccess(node0Handler));
  CmiSyncSendAndFree(0, CpvAccess(msgSize), msg);
}

void ringFinished(char *msg)
{
  // CmiPrintf("PE %d ringFinished\n",CmiMyPe());
  CmiFree(msg);
  CpvAccess(endTime) = CmiWallTimer();
  CpvAccess(endCTime) = CmiTimer();
  CmiPrintf("WALL: %4d \t%8d \t%8.4g \t%8.4g\n",
	    nCycles, CpvAccess(msgSize)-CmiMsgHeaderSizeBytes,
	    (CpvAccess(endTime)-CpvAccess(startTime))*1e3,
	    1e6*(CpvAccess(endTime)-CpvAccess(startTime))/(2.*nCycles)
  );
  CmiPrintf(" CPU: %4d \t%8d \t%8.4g \t%8.4g\n",
  	    nCycles,CpvAccess(msgSize)-CmiMsgHeaderSizeBytes,
 	    (CpvAccess(endCTime)-CpvAccess(startCTime))*1e3, 
  	    1e6*(CpvAccess(endCTime)-CpvAccess(startCTime))/(2.*nCycles)
    );
  CpvAccess(sizeNum)++;
  if (CpvAccess(msgSize) < maxMsgSize)
    startRing();
  else 
  {
    void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(sendmsg,CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
  }
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

  if (CpvAccess(cycleNum) == nCycles)
    ringFinished(msg);
  else
  {
    CmiSetHandler(msg,CpvAccess(node1Handler));
    CmiSyncSendAndFree(1,CpvAccess(msgSize),msg);
  }
  return 0;
}

CmiHandler node1HandlerFunc(char *msg)
{
  CpvAccess(msgSize) = *((int *)(msg+CmiMsgHeaderSizeBytes));
  CmiSetHandler(msg,CpvAccess(node0Handler));
  CmiSyncSendAndFree(0,CpvAccess(msgSize),msg);
  return 0;
}

CmiStartFn mymain()
{
  CmiPrintf("PE %d of %d starting\n",CmiMyPe(),CmiNumPes());

  CpvInitialize(int,msgSize);
  CpvInitialize(int,cycleNum);
  CpvInitialize(int,sizeNum);
  CpvAccess(sizeNum) = 1;

  CpvInitialize(int,exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
  CpvInitialize(int,node0Handler);
  CpvAccess(node0Handler) = CmiRegisterHandler((CmiHandler) node0HandlerFunc);
  CpvInitialize(int,node1Handler);
  CpvAccess(node1Handler) = CmiRegisterHandler((CmiHandler) node1HandlerFunc);

  CpvInitialize(double,startTime);
  CpvInitialize(double,endTime);
  CpvInitialize(double,startCTime);
  CpvInitialize(double,endCTime);

  if (CmiMyPe() == 0)
    startRing();

  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
  return 0;
}
