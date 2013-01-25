
/**********************************************************
      Converse Ping-pong to test the message latency and bandwidth
      Modified from Milind's ping-pong
      
      Sameer Kumar 02/07/05
***************************************************/

#include <stdlib.h>
#include <converse.h>

enum {nCycles = 1000};
enum { maxMsgSize = 1 << 14 };

CpvDeclare(int,msgSize);
CpvDeclare(int,cycleNum);
CpvDeclare(int,exitHandler);
CpvDeclare(int,node0Handler);
CpvDeclare(int,node1Handler);
CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);

#define USE_PERSISTENT     0

#if USE_PERSISTENT
PersistentHandle h;
#endif

// Start the pingpong for each message size
void startRing()
{
  CpvAccess(cycleNum) = 0;

  //Increase message in powers of 4. Also add a converse header to that
  CpvAccess(msgSize) = (CpvAccess(msgSize)-CmiMsgHeaderSizeBytes)*2 + 
      CmiMsgHeaderSizeBytes;

  char *msg = (char *)CmiAlloc(CpvAccess(msgSize));
  *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
  CmiSetHandler(msg,CpvAccess(node0Handler));
  CmiSyncSendAndFree(0, CpvAccess(msgSize), msg);
  CpvAccess(startTime) = CmiWallTimer();
}

//the pingpong has finished, record message time
void ringFinished(char *msg)
{
  CmiFree(msg);

  //Print the time for that message size
  CmiPrintf("Size=%d bytes, time=%lf microseconds one-way\n", 
             CpvAccess(msgSize)-CmiMsgHeaderSizeBytes, 
	     (1e6*(CpvAccess(endTime)-CpvAccess(startTime)))/(2.*nCycles));

  
  //Have we finished all message sizes?
  if (CpvAccess(msgSize) < maxMsgSize)
      //start the ring again
      startRing();
  else {
      //exit
      void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(sendmsg,CpvAccess(exitHandler));
      CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
  }
}

//We finished for all message sizes. Exit now
CmiHandler exitHandlerFunc(char *msg)
{
    CmiFree(msg);
    CsdExitScheduler();
    return 0;
}


//Handler on Node 0
CmiHandler node0HandlerFunc(char *msg)
{
    CpvAccess(cycleNum)++;
    
    if (CpvAccess(cycleNum) == nCycles) {
        CpvAccess(endTime) = CmiWallTimer();
        ringFinished(msg);
    }
    else {
        CmiSetHandler(msg,CpvAccess(node1Handler));
        *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
        
#if USE_PERSISTENT
        CmiUsePersistentHandle(&h, 1);
#endif
        CmiSyncSendAndFree(1,CpvAccess(msgSize),msg);
#if USE_PERSISTENT
        CmiUsePersistentHandle(NULL, 0);
#endif
    }
    return 0;
}

CmiHandler node1HandlerFunc(char *msg)
{
    CpvAccess(msgSize) = *((int *)(msg+CmiMsgHeaderSizeBytes));
    CmiSetHandler(msg,CpvAccess(node0Handler));
    
#if USE_PERSISTENT
    CmiUsePersistentHandle(&h, 1);
#endif
    CmiSyncSendAndFree(0,CpvAccess(msgSize),msg);
#if USE_PERSISTENT
    CmiUsePersistentHandle(NULL, 0);
#endif
    return 0;
}


//Converse main. Initialize variables and register handlers
CmiStartFn mymain()
{
    CpvInitialize(int,msgSize);
    CpvInitialize(int,cycleNum);
    
    CpvAccess(msgSize)= 512 + CmiMsgHeaderSizeBytes;
    
    CpvInitialize(int,exitHandler);
    CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
    CpvInitialize(int,node0Handler);
    CpvAccess(node0Handler) = CmiRegisterHandler((CmiHandler) node0HandlerFunc);
    CpvInitialize(int,node1Handler);
    CpvAccess(node1Handler) = CmiRegisterHandler((CmiHandler) node1HandlerFunc);
    
    CpvInitialize(double,startTime);
    CpvInitialize(double,endTime);
    
    int otherPe = CmiMyPe() ^ 1;
    
#if USE_PERSISTENT
    if (CmiMyPe() < CmiNumPes())
        h = CmiCreateCompressPersistent(otherPe, maxMsgSize+1024, 200, CMI_FLOATING);
#endif
    
    if (CmiMyPe() == 0)
        startRing();
    
    return 0;
}

int main(int argc,char *argv[])
{
    ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
    return 0;
}
