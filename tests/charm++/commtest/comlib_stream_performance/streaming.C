#include "comlib.h"
#include <cassert>

#include "streaming.decl.h"
CProxy_Main mainProxy;
int nElements;

CProxy_WorkerArray basicArrayProxy; 
CProxy_WorkerArray streamingArrayProxy;
ComlibInstanceHandle stratStreaming;


#define PERIOD_IN_MS 10
#define NMSGS 16
#define MAX_MESSAGE_SIZE 10000
#define MAX_BUFFER_SIZE  100000
#define ENVELOPE_OVERHEAD_ESTIMATE 100
#define MIN_TEST_SIZE 16
#define MAX_TEST_SIZE 1024

class TestMessage : public CMessage_TestMessage {
public:
  int length;
  char* msg;
};

// mainchare

class Main : public CBase_Main{
private:
  int nDone;

public:

  Main(CkArgMsg *m) {    
    nDone = 0;

    //    com_debug = 1;

    nElements = 2; 
    if(m->argc >1) nElements=atoi(m->argv[1]);
    delete m;

    mainProxy = thishandle;
	
    // create streaming strategy
    StreamingStrategy *strategy = new StreamingStrategy(PERIOD_IN_MS, NMSGS,
                                                        MAX_MESSAGE_SIZE, MAX_BUFFER_SIZE);
    stratStreaming = ComlibRegister(strategy);
    streamingArrayProxy = CProxy_WorkerArray::ckNew(nElements, nElements);
    basicArrayProxy = streamingArrayProxy; 

    ComlibAssociateProxy(stratStreaming, streamingArrayProxy);
    
    // initiate using non-delegated proxy because broadcasts do not
    // work with streaming
    basicArrayProxy.prepareTest();
    CkCallback syncWorkers(CkIndex_Main::runWithoutStreaming(), mainProxy); 
    CkStartQD(syncWorkers);   
  }

  void runWithStreaming() {
    CkPrintf("Running Streaming Test...\n"); 
    basicArrayProxy.initiateSends(streamingArrayProxy);
    CkCallback syncWorkers(CkIndex_Main::done(), mainProxy); 
    CkStartQD(syncWorkers);
  }

  void runWithoutStreaming() {
    CkPrintf("Running Without Streaming Enabled ... \n");
    basicArrayProxy.initiateSends(basicArrayProxy);
    CkCallback syncWorkers(CkIndex_Main::runWithStreaming(), mainProxy); 
    CkStartQD(syncWorkers);
  }

  void done() {
    CkPrintf("Finished test\n");
    CkExit();
  }

};

class WorkerArray : public CBase_WorkerArray {
private:
  CProxy_WorkerArray localProxy;
  TestMessage **msgs; 
  TestMessage **newMsgs; 
  int msgSize;
  int nElements; 
  int neighbor; 
  double mystartTime; 
  double myendTime;
  int receivedMsgs; 

public:

  WorkerArray(int nChares) {
    nElements = nChares; 
    msgs = new TestMessage*[NMSGS];
    newMsgs = new TestMessage*[NMSGS];
    msgSize = MIN_TEST_SIZE; 
    // partition into pairs of ranks
    if (thisIndex % 2 == 0) {
      neighbor = (thisIndex + nElements/2) % nElements; 
    }
    else {
      neighbor = (thisIndex - nElements/2) % nElements; 
    }
    receivedMsgs = 0; 
  }

  WorkerArray(CkMigrateMessage *m) {}

  void prepareTest() {
    
    for (int i = 0; i < NMSGS; i++) {
      msgs[i] = new(msgSize) TestMessage; 
    } 
        
  }


  void initiateSends(CProxy_WorkerArray workerProxy) {
    double startTime = CkWallTimer();
    localProxy = workerProxy; 
    if (thisIndex % 2 == 0) {
      mystartTime = CkWallTimer();    
      for (int i = 0; i < NMSGS; i++) {
        localProxy[neighbor].receiveSends(msgs[i]);
      }
    }
    double endTime = CkWallTimer(); 
    //    CkPrintf("[%d] initiateSends took %f us\n", thisIndex, (endTime-startTime) * 1000000);
  }

  void receiveSends(TestMessage *msg) {
    double startTime = CkWallTimer();
    // recycle received messages
    newMsgs[receivedMsgs] = msg; 
    receivedMsgs++; 
    if (receivedMsgs == NMSGS) {
      for (int i=0; i < NMSGS; i++) {
        localProxy[neighbor].receiveReplies(msgs[i]); 
      }
      receivedMsgs = 0;
      msgs = newMsgs; 
    }
    double endTime = CkWallTimer(); 
    //    CkPrintf("[%d] receiveSends took %f us\n", thisIndex, (endTime-startTime) * 1000000);
  }
  
  void receiveReplies(TestMessage *msg) {
    double startTime = CkWallTimer();
    // recycle received messages
    newMsgs[receivedMsgs] = msg;
    receivedMsgs++;
    if (receivedMsgs == NMSGS) {
      myendTime = CkWallTimer();
      CkPrintf("[%d] round trip time for sending %d messages: %f us\n", 
               thisIndex, NMSGS, 1000000 * (myendTime - mystartTime));
      receivedMsgs = 0;
      msgs = newMsgs; 
    }
    double endTime = CkWallTimer(); 
    //    CkPrintf("[%d] receiveReplies took %f us\n", thisIndex, (endTime-startTime) * 1000000);
  }

};


#include "streaming.def.h"
