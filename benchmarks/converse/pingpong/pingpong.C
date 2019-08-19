
/***************************************************************
  Converse Ping-pong to test the message latency and bandwidth
  Modified from Milind's ping-pong

  Sameer Kumar 02/07/05
 ****************************************************************/

#include <stdlib.h>
#include <converse.h>

CpvDeclare(int,nCycles);
CpvDeclare(int,minMsgSize);
CpvDeclare(int,maxMsgSize);
CpvDeclare(int,factor);
CpvDeclare(bool,warmUp);
CpvDeclare(int,msgSize);
CpvDeclare(int,cycleNum);
CpvDeclare(int,warmUpDoneHandler);
CpvDeclare(int,exitHandler);
CpvDeclare(int,node0Handler);
CpvDeclare(int,node1Handler);
CpvDeclare(int,startOperationHandler);
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
  char *msg = (char *)CmiAlloc(CpvAccess(msgSize));
  *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
  CmiSetHandler(msg,CpvAccess(node0Handler));
  CmiSyncSendAndFree(0, CpvAccess(msgSize), msg);
}

//the pingpong has finished, record message time
void ringFinished(char *msg)
{
  size_t msgSizeDiff = CpvAccess(msgSize)-CmiMsgHeaderSizeBytes;
  CmiFree(msg);

  //Print the time for that message size
  CmiPrintf("Size=%zu bytes, time=%lf microseconds one-way\n",
      msgSizeDiff,
      (1e6*(CpvAccess(endTime)-CpvAccess(startTime)))/(2.*CpvAccess(nCycles)));


  //Have we finished all message sizes?
  if ((CpvAccess(msgSize) - CmiMsgHeaderSizeBytes) < CpvAccess(maxMsgSize)) {
    //Increase message in powers of factor. Also add a converse header to that
    CpvAccess(msgSize) = (CpvAccess(msgSize)-CmiMsgHeaderSizeBytes)*CpvAccess(factor) +
      CmiMsgHeaderSizeBytes;

    //start the ring again
    startRing();
  }
  else {
    //exit
    void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(sendmsg,CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
  }
}

void startWarmUp()
{
  // Small pingpong message to ensure that setup is completed
  char *msg = (char *)CmiAlloc(CpvAccess(msgSize));
  *((int *)(msg+CmiMsgHeaderSizeBytes)) = CpvAccess(msgSize);
  CmiSetHandler(msg,CpvAccess(node0Handler));
  CmiSyncSendAndFree(0, CpvAccess(msgSize), msg);
}

// Handler on Node 0 which starts pingpong on warmup completion
void warmUpDoneHandlerFunc(char *msg)
{
  CmiFree(msg);
  // Warmup phase completed. Start pingpong
  startRing();
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
  if(CpvAccess(warmUp))
    CpvAccess(warmUp) = false;
  else
    CpvAccess(cycleNum)++;

  // Begin timer for the first iteration
  if(CpvAccess(cycleNum) == 1)
    CpvAccess(startTime) = CmiWallTimer();

  // Stop timer for the last iteration
  if (CpvAccess(cycleNum) == CpvAccess(nCycles)) {
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

  if(CpvAccess(warmUp)) {
    CmiSetHandler(msg, CpvAccess(warmUpDoneHandler));
    CpvAccess(warmUp) = false;
  }
  else
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

// Converse handler for beginning operation
CmiHandler startOperationHandlerFunc(char *msg)
{
#if USE_PERSISTENT
  if (CmiMyPe() < CmiNumPes())
    h = CmiCreateCompressPersistent(otherPe, CpvAccess(maxMsgSize)+1024, 200, CMI_FLOATING);
#endif

  if (CmiMyPe() == 0)
    startWarmUp();
  return 0;
}

//Converse main. Initialize variables and register handlers
CmiStartFn mymain(int argc, char *argv[])
{
  CpvInitialize(int,msgSize);
  CpvInitialize(int,cycleNum);

  CpvInitialize(int,nCycles);
  CpvInitialize(int,minMsgSize);
  CpvInitialize(int,maxMsgSize);
  CpvInitialize(int,factor);
  CpvInitialize(bool,warmUp);

  // Register Handlers
  CpvInitialize(int,warmUpDoneHandler);
  CpvAccess(warmUpDoneHandler) = CmiRegisterHandler((CmiHandler) warmUpDoneHandlerFunc);
  CpvInitialize(int,exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
  CpvInitialize(int,node0Handler);
  CpvAccess(node0Handler) = CmiRegisterHandler((CmiHandler) node0HandlerFunc);
  CpvInitialize(int,node1Handler);
  CpvAccess(node1Handler) = CmiRegisterHandler((CmiHandler) node1HandlerFunc);
  CpvInitialize(int,startOperationHandler);
  CpvAccess(startOperationHandler) = CmiRegisterHandler((CmiHandler) startOperationHandlerFunc);

  //set warmup run
  CpvAccess(warmUp) = true;

  CpvInitialize(double,startTime);
  CpvInitialize(double,endTime);

  int otherPe = CmiMyPe() ^ 1;

  // Set runtime cpuaffinity
  CmiInitCPUAffinity(argv);

  // Initialize CPU topology
  CmiInitCPUTopology(argv);

  // Wait for all PEs of the node to complete topology init
  CmiNodeAllBarrier();

  // Update the argc after runtime parameters are extracted out
  argc = CmiGetArgc(argv);
  if(argc == 5){
    CpvAccess(nCycles)=atoi(argv[1]);
    CpvAccess(minMsgSize)= atoi(argv[2]);
    CpvAccess(maxMsgSize)= atoi(argv[3]);
    CpvAccess(factor)= atoi(argv[4]);
  } else if(argc == 1) {
    // use default arguments
    CpvAccess(nCycles) = 1000;
    CpvAccess(minMsgSize) = 1 << 9;
    CpvAccess(maxMsgSize) = 1 << 14;
    CpvAccess(factor) = 2;
  } else {
    CmiAbort("Usage: ./pingpong <ncycles> <minsize> <maxsize> <increase factor> \nExample: ./pingpong 100 2 128 2\n");
  }

  if(CmiMyPe() == 0) {
    CmiPrintf("Pingpong with iterations = %d, minMsgSize = %d, maxMsgSize = %d, increase factor = %d\n",
        CpvAccess(nCycles), CpvAccess(minMsgSize), CpvAccess(maxMsgSize), CpvAccess(factor));
  }

  CpvAccess(msgSize)= CpvAccess(minMsgSize) + CmiMsgHeaderSizeBytes;

  // Node 0 waits till all processors finish their topology processing
  if(CmiMyPe() == 0) {
    // Signal all PEs to begin computing
    char *startOperationMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler((char *)startOperationMsg, CpvAccess(startOperationHandler));
    CmiSyncBroadcastAndFree(CmiMsgHeaderSizeBytes, startOperationMsg);

    // start operation locally on PE 0
    startOperationHandlerFunc(NULL);
  }
  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
  return 0;
}
