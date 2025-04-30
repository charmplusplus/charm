
/***************************************************************
  Converse self send benchmark
 ****************************************************************/

#include <converse.h>
#include <cstdlib>
#include <ctime>
#include <vector>

CpvDeclare(int, iter);
CpvDeclare(int, msgSize);

int iterations;

CpvDeclare(int, acksReported);

CpvDeclare(int, exitHandler);
CpvDeclare(int, completionHandler);
CpvDeclare(int, initiateHandler);
CpvDeclare(int, ackMsgHandler);
CpvDeclare(int, selfSendMsgHandler);

CpvDeclare(int, compCounter);

CpvStaticDeclare(double, startTime);
CpvStaticDeclare(double, endTime);

struct selfSendMsg {
  char core[CmiMsgHeaderSizeBytes];
  int srcPe;
};

void initiate() {

  if (CpvAccess(iter) == iterations) {

    char *compMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(compMsg, CpvAccess(completionHandler));
    CmiSyncSendAndFree(0, CmiMsgHeaderSizeBytes, compMsg);

  } else {

    CpvAccess(iter) = CpvAccess(iter) + 1;

    if (CmiMyPe() == 0)
      CmiPrintf("Running iteration %d\n", CpvAccess(iter));

    // Send a message to myself
    int totalSize = sizeof(selfSendMsg) + CpvAccess(msgSize);
    selfSendMsg *msg = (selfSendMsg *)CmiAlloc(totalSize);
    msg->srcPe = CmiMyPe();
    CmiSetHandler(msg, CpvAccess(selfSendMsgHandler));

    CmiSyncSend(CmiMyPe(), totalSize, msg);
    CmiFree(msg);
  }
}

void handleCompletion(char *msg) {
  CpvAccess(compCounter)++;

  if (CpvAccess(compCounter) == CmiNumPes()) {
    CpvAccess(endTime) = CmiWallTimer();
    CmiPrintf("[%d][%d][%d] self send all iterations completed, time taken is "
              "%lf s\n",
              CmiMyPe(), CmiMyNode(), CmiMyRank(),
              (CpvAccess(endTime) - CpvAccess(startTime)));
    CmiSetHandler(msg, CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, msg);
  } else {
    CmiFree(msg);
  }
}

void handleInitiate(char *initiateMsg) {
  CmiFree(initiateMsg);
  initiate();
}

void handleExit(char *msg) {
  CmiFree(msg);
  CsdExitScheduler();
}

void handleSelfSend(selfSendMsg *msg) {
  char *ackMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(ackMsg, CpvAccess(ackMsgHandler));
  CmiSyncSendAndFree(msg->srcPe, CmiMsgHeaderSizeBytes, ackMsg);
}

void handleAck(char *msg) {

  CmiFree(msg);
  initiate();
}

// Converse main. Initialize variables and register handlers
CmiStartFn mymain(int argc, char *argv[]) {

  // iterations value
  CpvInitialize(int, iter);

  // msg size value
  CpvInitialize(int, msgSize);

  // Register Handlers
  CpvInitialize(int, initiateHandler);
  CpvAccess(initiateHandler) = CmiRegisterHandler((CmiHandler)handleInitiate);

  CpvInitialize(int, completionHandler);
  CpvAccess(completionHandler) =
      CmiRegisterHandler((CmiHandler)handleCompletion);

  CpvInitialize(int, selfSendMsgHandler);
  CpvAccess(selfSendMsgHandler) =
      CmiRegisterHandler((CmiHandler)handleSelfSend);

  CpvInitialize(int, ackMsgHandler);
  CpvAccess(ackMsgHandler) = CmiRegisterHandler((CmiHandler)handleAck);

  CpvInitialize(int, exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler)handleExit);

  CpvInitialize(double, startTime);
  CpvInitialize(double, endTime);

  // Set runtime cpuaffinity
//   CmiInitCPUAffinity(argv);

  // Initialize CPU topology
//   CmiInitCPUTopology(argv);

  // Wait for all PEs of the node to complete topology init
  CmiNodeAllBarrier();

  // Update the argc after runtime parameters are extracted out
  argc = CmiGetArgc(argv);

  if (CmiMyPe() == 0) {
    CmiPrintf("argc: %d\n", argc);
  }

  if (argc == 3) {
    iterations = atoi(argv[1]);
    CpvAccess(msgSize) = atoi(argv[2]);

  } else if (argc == 1) {
    iterations = 100;
    CpvAccess(msgSize) = 1000;

  } else {

    if (CmiMyPe() == 0)
      CmiAbort("Usage: ./self_send <iter> <msgSize>\n");
  }

  CpvInitialize(int, compCounter);
  CpvAccess(compCounter) = 0;

  CpvInitialize(int, acksReported);
  CpvAccess(acksReported) = 0;

  CpvAccess(iter) = 0;

  if (CmiMyPe() == 0) {
    CmiPrintf("Launching self_send with iterations=%d, msgSize=%d\n",
              iterations, CpvAccess(msgSize));
  }

    CpvAccess(startTime) = CmiWallTimer();
    char *initiateMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(initiateMsg, CpvAccess(initiateHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, initiateMsg);
  return 0;
}

int main(int argc, char *argv[]) {
  ConverseInit(argc, argv, (CmiStartFn)mymain, 0, 0);
  return 0;
}
