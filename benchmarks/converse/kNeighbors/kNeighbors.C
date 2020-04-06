
/***************************************************************
  Converse kNeighbors benchmark
 ****************************************************************/

#include <converse.h>
#include <vector>
#include <ctime>
#include <cstdlib>

CpvDeclare(int, k);
CpvDeclare(int, iter);
CpvDeclare(int, msgSize);

int iterations;

CpvDeclare(int, acksReported);

CpvDeclare(int, exitHandler);
CpvDeclare(int, completionHandler);
CpvDeclare(int, initiateHandler);
CpvDeclare(int, ackMsgHandler);
CpvDeclare(int, kNeighborMsgHandler);

CpvDeclare(int, compCounter);

CpvDeclare(std::vector<int>, neighbors);

CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);

struct kNeighborMsg {
  char core[CmiMsgHeaderSizeBytes];
  int srcPe;
};

void initiate() {

  if(CpvAccess(iter) == iterations) {

    char *compMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(compMsg, CpvAccess(completionHandler));
    CmiSyncSendAndFree(0, CmiMsgHeaderSizeBytes, compMsg);

  } else {

    CpvAccess(iter) = CpvAccess(iter) + 1;

    if(CmiMyPe() == 0)
      CmiPrintf("Running iteration %d\n", CpvAccess(iter));

    // Send a message to all my kneighbors
    int totalSize = sizeof(kNeighborMsg) + CpvAccess(msgSize);
    kNeighborMsg *msg = (kNeighborMsg *)CmiAlloc(totalSize);
    msg->srcPe = CmiMyPe();
    CmiSetHandler(msg, CpvAccess(kNeighborMsgHandler));

    for (int i=0; i < CpvAccess(neighbors).size(); i++) {
      int neighborPe = CpvAccess(neighbors).at(i);
      CmiSyncSend(neighborPe, totalSize, msg);
    }
    CmiFree(msg);
  }
}

// Called on all PEs (from the kNeighbor PEs)
void handleCompletion(char *msg) {
  CpvAccess(compCounter)++;

  if(CpvAccess(compCounter) == CmiNumPes()) {
    CpvAccess(endTime) = CmiWallTimer();
    CmiPrintf("[%d][%d][%d] kNeighbors all iterations completed, time taken is %lf s\n", CmiMyPe(), CmiMyNode(),
                                    CmiMyRank(), (CpvAccess(endTime) - CpvAccess(startTime)));
    CmiSetHandler(msg, CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, msg);
  } else {
    CmiFree(msg);
  }
}

// Called on all PEs to initiate kNeighbors
void handleInitiate(char *initiateMsg) {
  CmiFree(initiateMsg);
  initiate();
}

// Called on all PEs to exit
void handleExit(char *msg) {
  CmiFree(msg);
  CsdExitScheduler();
}

// Called on all kNeighbor PEs
void handleNeighborMsg(kNeighborMsg *msg) {
  // Send a message back to the source pe notifying that the message has been received
  char *ackMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(ackMsg, CpvAccess(ackMsgHandler));
  CmiSyncSendAndFree(msg->srcPe, CmiMsgHeaderSizeBytes, ackMsg);
}

void handleAck(char *msg) {

  CmiFree(msg);
  CpvAccess(acksReported) = CpvAccess(acksReported) + 1;

  if(CpvAccess(acksReported) == CpvAccess(k)) {

    CpvAccess(acksReported) = 0;

    // Iteration complete, start next iteration
    initiate();
  }
}

//Converse main. Initialize variables and register handlers
CmiStartFn mymain(int argc, char *argv[])
{

  // k value
  CpvInitialize(int, k);

  // iterations value
  CpvInitialize(int, iter);

  // msg size value
  CpvInitialize(int, msgSize);

  // Register Handlers
  CpvInitialize(int, initiateHandler);
  CpvAccess(initiateHandler) = CmiRegisterHandler((CmiHandler) handleInitiate);

  CpvInitialize(int, completionHandler);
  CpvAccess(completionHandler) = CmiRegisterHandler((CmiHandler) handleCompletion);

  CpvInitialize(int, kNeighborMsgHandler);
  CpvAccess(kNeighborMsgHandler) = CmiRegisterHandler((CmiHandler) handleNeighborMsg);

  CpvInitialize(int, ackMsgHandler);
  CpvAccess(ackMsgHandler) = CmiRegisterHandler((CmiHandler) handleAck);

  CpvInitialize(int, exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) handleExit);

  CpvInitialize(double,startTime);
  CpvInitialize(double,endTime);

  // Set runtime cpuaffinity
  CmiInitCPUAffinity(argv);

  // Initialize CPU topology
  CmiInitCPUTopology(argv);

  // Wait for all PEs of the node to complete topology init
  CmiNodeAllBarrier();

  // Update the argc after runtime parameters are extracted out
  argc = CmiGetArgc(argv);

  if(argc == 4){

    CpvAccess(k) = atoi(argv[1]);
    iterations = atoi(argv[2]);
    CpvAccess(msgSize) = atoi(argv[3]);

  } else if(argc == 1) {

    CpvAccess(k) = CmiNumPes()/2;
    iterations = 100;
    CpvAccess(msgSize) = 1000;

  } else {

    if(CmiMyPe() == 0)
      CmiAbort("Usage: ./kNeighbors <k> <iter> <msgSize>\n");
  }

  if(CmiMyPe() == 0) {
    if(CpvAccess(k) > CmiNumPes())
      CmiAbort("k cannot be greater than the number of Pes\n");
  }

  int lowVal, highVal;

  // Determine my neighbor pes
  if(CpvAccess(k) % 2 == 0) { // choose k/2 left neighbors and k/2 right neighbors

    lowVal = CpvAccess(k)/2;
    highVal = CpvAccess(k)/2;

  } else { // choose k/2 left neighbors and k/2 + 1 right neighbors

    lowVal = CpvAccess(k)/2;
    highVal = CpvAccess(k)/2 + 1;

  }

  CpvInitialize(int, compCounter);
  CpvAccess(compCounter) = 0;

  CpvInitialize(int, acksReported);
  CpvAccess(acksReported) = 0;

  for(int i = CmiMyPe() - lowVal; i <= CmiMyPe() + highVal; i++) {
    if(i == CmiMyPe()) continue;
    if(i < 0) CpvAccess(neighbors).push_back(i + CmiNumPes());
    else if(i > CmiNumPes() - 1) CpvAccess(neighbors).push_back(i - CmiNumPes());
    else CpvAccess(neighbors).push_back(i);
  }

  CpvAccess(iter) = 0;

  if(CmiMyPe() == 0) {
    CmiPrintf("Launching kNeighbors with k=%d, iterations=%d, msgSize=%d\n", CpvAccess(k), iterations, CpvAccess(msgSize));

    CpvAccess(startTime) = CmiWallTimer();
    char *initiateMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(initiateMsg, CpvAccess(initiateHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, initiateMsg);
  }
  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
  return 0;
}
