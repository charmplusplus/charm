
/***************************************************************
  Converse Random TTL benchmark
 ****************************************************************/

#include <converse.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <iterator>

CpvDeclare(int, k);
CpvDeclare(int, ttl);

CpvDeclare(int, compCounter);

CpvDeclare(int, ttlMsgHandler);
CpvDeclare(int, completionHandler);
CpvDeclare(int, exitHandler);
CpvDeclare(int, initiateHandler);

CpvDeclare(std::vector<int>, peVector);

CpvStaticDeclare(double,startTime);
CpvStaticDeclare(double,endTime);

struct ttlMsg {
  char core[CmiMsgHeaderSizeBytes];
  int ttlVal;
};

std::random_device rd;
std::mt19937 gen(rd());

// Method to get a random value from the peVector
int getRandomKFromPeVector() {
  CmiAssert(CpvAccess(peVector).size() == CmiNumPes());
  std::vector<int>::iterator start = CpvAccess(peVector).begin();
  std::vector<int>::iterator end = CpvAccess(peVector).end();

  std::uniform_int_distribution<> distribution(0, std::distance(start, end) - 1);
  std::advance(start, distribution(gen));
  return *start;
}

// Called on all PEs to initiate TTL
void handleInitiate(char *initiateMsg) {

  CmiFree(initiateMsg);

  // Create a message to hold an integer
  ttlMsg *msg = (ttlMsg *)CmiAlloc(sizeof(ttlMsg));
  msg->ttlVal = CpvAccess(ttl);
  CmiSetHandler(msg, CpvAccess(ttlMsgHandler));


  // send messages to random k pes
  for (int i=0; i < CpvAccess(k); i++) {
    CmiSyncSend(i, sizeof(ttlMsg), msg);
  }

  CmiFree(msg);
}

// Called on different PEs based on the random numbers generated
void handleTtlMessage(ttlMsg *msg) {

  msg->ttlVal -= 1; // Decrement by 1

  if(msg->ttlVal == 0) {
    // this ttl thread is complete
    // send the same received message to 0 with the completion handler to signal completion
    CmiSetHandler(msg, CpvAccess(completionHandler));
    CmiSyncSendAndFree(0, sizeof(ttlMsg), msg);

  } else {
    // select a random PE and send the same message to it
    int k = getRandomKFromPeVector();
    CmiSyncSendAndFree(k, sizeof(ttlMsg), msg);
  }
}

// Called on PE 0
void handleCompletion(ttlMsg *msg) {
  CpvAccess(compCounter)++;

  if(CpvAccess(compCounter) == CpvAccess(k) * CmiNumPes()) { // All launched ttls are complete

    CpvAccess(endTime) = CmiWallTimer();
    CmiPrintf("[%d][%d][%d] Random TTL Completed, time taken is %lf s\n", CmiMyPe(), CmiMyNode(),
                                                  CmiMyRank(), (CpvAccess(endTime)-CpvAccess(startTime)));
    // Exit, reuse msg
    CmiSetHandler(msg, CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(sizeof(ttlMsg), msg);

  } else {
    CmiFree(msg);
  }
}

// Called on all PEs
void handleExit(ttlMsg *msg)
{
  CmiFree(msg);
  // Exit
  CsdExitScheduler();
}

//Converse main. Initialize variables and register handlers
CmiStartFn mymain(int argc, char *argv[])
{

  // k value
  CpvInitialize(int, k);

  // TTL value
  CpvInitialize(int, ttl);

  // Completion counter
  CpvInitialize(int, compCounter);
  CpvAccess(compCounter) = 0;

  // Register Handlers
  CpvInitialize(int, initiateHandler);
  CpvAccess(initiateHandler) = CmiRegisterHandler((CmiHandler) handleInitiate);

  CpvInitialize(int, ttlMsgHandler);
  CpvAccess(ttlMsgHandler) = CmiRegisterHandler((CmiHandler) handleTtlMessage);

  CpvInitialize(int, completionHandler);
  CpvAccess(completionHandler) = CmiRegisterHandler((CmiHandler) handleCompletion);

  CpvInitialize(int, exitHandler);
  CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) handleExit);

  CpvInitialize(std::vector<int>, peVector);

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

  if(argc == 3){
    CpvAccess(k) = atoi(argv[1]);
    CpvAccess(ttl) = atoi(argv[2]);
  } else if(argc == 1) {
    CpvAccess(k) = CmiNumPes()/2;
    CpvAccess(ttl) = 1000;
  } else {
    if(CmiMyPe() == 0)
      CmiAbort("Usage: ./randomttl <k> <ttl>\n");
  }

  std::srand (unsigned(std::time(0)) + CmiMyPe());

  // set values to Pes
  CpvAccess(peVector).resize(CmiNumPes());
  std::iota(CpvAccess(peVector).begin(), CpvAccess(peVector).end(), 0);

  // using built-in random generator
  std::random_shuffle(CpvAccess(peVector).begin(), CpvAccess(peVector).end());

  if(CmiMyPe() == 0) {
    if(CpvAccess(k) > CmiNumPes())
      CmiAbort("k cannot be greater than the number of Pes\n");

    CmiPrintf("Launching Random TTL with k=%d, ttl=%d\n", CpvAccess(k), CpvAccess(ttl));

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
