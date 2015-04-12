#include "overhead_test.decl.h"
#include <vector>

// This benchmark measures communication overhead and bandwidth for
//  Charm++ groups and arrays, in similar fashion to the converse
//  benchmark: tests/converse/machinetest/multiping.C

// Index 0 sends kFactor messages to Index 1 and gets an ack for them.
//  The benchmark measures bandwidth for this burst of messages.
int kFactor = 64;
int minMsgSize = 16;
int maxMsgSize = 262144;
int nCycles = 100; // Number of iterations for each message size

CProxy_TestDriver testDriverProxy;

void idleStartHandler(void *timingGroupObj, double start);
void idleEndHandler(void *timingGroupObj, double cur);
void idleStartHandlerArray(void *timingGroupObj, double start);
void idleEndHandlerArray(void *timingGroupObj, double cur);

class SimpleMessage: public CMessage_SimpleMessage {
public:
  char *data;
};

class TestDriver: public CBase_TestDriver {
private:
  CProxy_CommunicationArray communicationArrayProxy;
  CProxy_CommunicationGroup communicationGroupProxy;
  bool doneGroupTest;
  bool timeAllocation;

public:
  TestDriver(CkArgMsg *args) {
    if (args->argc > 1) {
      kFactor = atoi(args->argv[1]);
      minMsgSize = atoi(args->argv[2]);
      maxMsgSize = atoi(args->argv[3]);
      nCycles = atoi(args->argv[4]);
    }

    communicationGroupProxy = CProxy_CommunicationGroup::ckNew();
    communicationArrayProxy = CProxy_CommunicationArray::ckNew(CkNumPes());

    doneGroupTest = false;
    timeAllocation = false;
    runTest();
  }

  void runTest() {
    if (!doneGroupTest) {
      CkPrintf("\nCharm++ group communication with allocation timing %s\n\n",
               timeAllocation ? "enabled" : "disabled");
      CkPrintf("%3s %20s %20s %20s %20s\n", "PE", "MSG SIZE", "PER MSG TIME(us)",
               "BW(MB/s)", "OVERHEAD(us)");
      communicationGroupProxy.startOperation(timeAllocation);
    }
    else {
      CkPrintf("\nCharm++ 1D array communication with allocation timing %s\n\n",
               timeAllocation ? "enabled" : "disabled");
      CkPrintf("%3s %20s %20s %20s %20s\n", "PE", "MSG SIZE", "PER MSG TIME(us)",
               "BW(MB/s)", "OVERHEAD(us)");
      communicationArrayProxy.startOperation(timeAllocation);
    }

  }

  void testDone() {
    if (doneGroupTest && timeAllocation == true) {
      CkExit();
    }
    else {
      if (timeAllocation == true) {
        doneGroupTest = true;
      }
      timeAllocation = !timeAllocation;
      runTest();
    }
  }

};

class CommunicationGroup: public CBase_CommunicationGroup {
private:
  int cycleNum;
  int msgSize;
  int neighbor;
  int nReceived;
  double startTime;
  double totalTime;

  std::vector<SimpleMessage *> sentMessages;
  std::vector<SimpleMessage *> receivedMessages;
  bool timeAllocation;
public:

  int beginHandlerId;
  int endHandlerId;

  double startIdleTime;
  double iterationIdleTime;
  double totalIdleTime;

  CommunicationGroup() {
    nReceived = 0;
    beginHandlerId = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
                                         idleStartHandler, (void *) this);
    endHandlerId = CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,
                                         idleEndHandler, (void *) this);
    msgSize = minMsgSize;
    cycleNum = 0;
    neighbor = (CkMyPe() + CkNumPes() / 2) % CkNumPes();
    totalTime = 0.0;
    totalIdleTime = 0.0;
  }

  void startOperation(bool includeAlloc) {
    timeAllocation = includeAlloc;
    iterationIdleTime = 0.0;
    if (CkMyPe() < CkNumPes() / 2) {
      if (timeAllocation) {
        startTime = CkWallTimer();
        for (int i = 0; i < kFactor; i++) {
          SimpleMessage *msg = new (msgSize) SimpleMessage();
          thisProxy[neighbor].receiveMessage(msg);
        }
      }
      else {
        for (int i = 0; i < kFactor; i++) {
          SimpleMessage *msg = new (msgSize) SimpleMessage();
          sentMessages.push_back(msg);
        }
        startTime = CkWallTimer();
        for (int i = 0; i < kFactor; i++) {
          thisProxy[neighbor].receiveMessage(sentMessages[i]);
        }
      }
    }
    else {
      startTime = CkWallTimer();
      receivedMessages.reserve(kFactor);
    }
  }

  void receiveMessage(SimpleMessage *msg) {
    if (timeAllocation) {
      delete msg;
      if (++nReceived == kFactor) {
        nReceived = 0;
        operationFinished(NULL);
        msg = new (msgSize) SimpleMessage();
        thisProxy[neighbor].operationFinished(msg);
      }
    }
    else {
      if (receivedMessages.size() == kFactor - 1) {
        thisProxy[neighbor].operationFinished(msg);
        operationFinished(NULL);
      }
      else {
        receivedMessages.push_back(msg);
      }
    }
  }

  void operationFinished(SimpleMessage *msg) {
    double endTime = CkWallTimer();
    totalTime += endTime - startTime;
    totalIdleTime += iterationIdleTime;
    cycleNum++;
    for (int i = 0; i < receivedMessages.size(); i++) {
      delete receivedMessages[i];
    }
    sentMessages.clear();
    receivedMessages.clear();

    if (cycleNum == nCycles) {
      double numIterations =
        msg == NULL ? nCycles * kFactor : nCycles * (kFactor + 1);
      delete msg;
      double cycleTime = 1e6 * totalTime / numIterations;
      double idleTimePerCycle = 1e6 * totalIdleTime / numIterations;
      double computeTime = cycleTime - idleTimePerCycle;
      double bandwidth = msgSize * 1e6 / cycleTime / 1024.0 / 1024.0;
      CkPrintf("[%d] %20d %20.3lf %20.3lf %20.3lf\n",
               CmiMyPe(), msgSize, cycleTime, bandwidth, computeTime);
      totalIdleTime = 0.0;
      totalTime = 0.0;
      msgSize *= 2;
      cycleNum = 0;
    }

    if (msgSize <= maxMsgSize) {
      startOperation(timeAllocation);
    }
    else {
      if (timeAllocation == true) {
        CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, beginHandlerId);
        CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE, endHandlerId);
      }
      msgSize = minMsgSize;
      cycleNum = 0;
      neighbor = (CkMyPe() + CkNumPes() / 2) % CkNumPes();
      totalTime = 0.0;
      totalIdleTime = 0.0;
      contribute(CkCallback(CkReductionTarget(TestDriver, testDone),
                            testDriverProxy));
    }
  }

};

// TO DO - remove code duplication - code is almost the same as group version
class CommunicationArray: public CBase_CommunicationArray {
private:
  int cycleNum;
  int msgSize;
  int neighbor;
  int nReceived;
  double startTime;
  double totalTime;

  std::vector<SimpleMessage *> sentMessages;
  std::vector<SimpleMessage *> receivedMessages;
  bool timeAllocation;
public:

  int beginHandlerId;
  int endHandlerId;

  double startIdleTime;
  double iterationIdleTime;
  double totalIdleTime;

  CommunicationArray() {
    nReceived = 0;
    beginHandlerId = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
                                         idleStartHandlerArray, (void *) this);
    endHandlerId = CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,
                                         idleEndHandlerArray, (void *) this);
    msgSize = minMsgSize;
    cycleNum = 0;
    neighbor = (CkMyPe() + CkNumPes() / 2) % CkNumPes();
    totalTime = 0.0;
    totalIdleTime = 0.0;
  }

  CommunicationArray(CkMigrateMessage *msg) {}

  void startOperation(bool includeAlloc) {
    timeAllocation = includeAlloc;
    iterationIdleTime = 0.0;
    if (CkMyPe() < CkNumPes() / 2) {
      if (timeAllocation) {
        startTime = CkWallTimer();
        for (int i = 0; i < kFactor; i++) {
          SimpleMessage *msg = new (msgSize) SimpleMessage();
          thisProxy[neighbor].receiveMessage(msg);
        }
      }
      else {
        for (int i = 0; i < kFactor; i++) {
          SimpleMessage *msg = new (msgSize) SimpleMessage();
          sentMessages.push_back(msg);
        }
        startTime = CkWallTimer();
        for (int i = 0; i < kFactor; i++) {
          thisProxy[neighbor].receiveMessage(sentMessages[i]);
        }
      }
    }
    else {
      startTime = CkWallTimer();
      receivedMessages.reserve(kFactor);
    }
  }

  void receiveMessage(SimpleMessage *msg) {
    if (timeAllocation) {
      delete msg;
      if (++nReceived == kFactor) {
        nReceived = 0;
        operationFinished(NULL);
        msg = new (msgSize) SimpleMessage();
        thisProxy[neighbor].operationFinished(msg);
      }
    }
    else {
      if (receivedMessages.size() == kFactor - 1) {
        thisProxy[neighbor].operationFinished(msg);
        operationFinished(NULL);
      }
      else {
        receivedMessages.push_back(msg);
      }
    }
  }

  void operationFinished(SimpleMessage *msg) {
    double endTime = CkWallTimer();
    totalTime += endTime - startTime;
    totalIdleTime += iterationIdleTime;
    cycleNum++;
    for (int i = 0; i < receivedMessages.size(); i++) {
      delete receivedMessages[i];
    }
    sentMessages.clear();
    receivedMessages.clear();

    if (cycleNum == nCycles) {
      double numIterations =
        msg == NULL ? nCycles * kFactor : nCycles * (kFactor + 1);
      delete msg;
      double cycleTime = 1e6 * totalTime / numIterations;
      double idleTimePerCycle = 1e6 * totalIdleTime / numIterations;
      double computeTime = cycleTime - idleTimePerCycle;
      double bandwidth = msgSize * 1e6 / cycleTime / 1024.0 / 1024.0;
      CkPrintf("[%d] %20d %20.3lf %20.3lf %20.3lf\n",
               CmiMyPe(), msgSize, cycleTime, bandwidth, computeTime);
      totalIdleTime = 0.0;
      totalTime = 0.0;
      msgSize *= 2;
      cycleNum = 0;
    }

    if (msgSize <= maxMsgSize) {
      startOperation(timeAllocation);
    }
    else {
      if (timeAllocation == true) {
        CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, beginHandlerId);
        CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE, endHandlerId);
      }
      msgSize = minMsgSize;
      cycleNum = 0;
      neighbor = (CkMyPe() + CkNumPes() / 2) % CkNumPes();
      totalTime = 0.0;
      totalIdleTime = 0.0;
      contribute(CkCallback(CkReductionTarget(TestDriver, testDone),
                            testDriverProxy));
    }
  }

};

void idleStartHandler(void *timingGroupObj, double start) {
  CommunicationGroup *localInstance = (CommunicationGroup *) timingGroupObj;
  localInstance->startIdleTime = start;
}

void idleEndHandler(void *timingGroupObj, double cur) {
  CommunicationGroup *localInstance = (CommunicationGroup *) timingGroupObj;
  if(localInstance->startIdleTime > 0) {
    localInstance->iterationIdleTime += cur - localInstance->startIdleTime;
    localInstance->startIdleTime = -1;
  }
}

void idleStartHandlerArray(void *timingGroupObj, double start) {
  CommunicationArray *localInstance = (CommunicationArray *) timingGroupObj;
  localInstance->startIdleTime = start;
}

void idleEndHandlerArray(void *timingGroupObj, double cur) {
  CommunicationArray *localInstance = (CommunicationArray *) timingGroupObj;
  if(localInstance->startIdleTime > 0) {
    localInstance->iterationIdleTime += cur - localInstance->startIdleTime;
    localInstance->startIdleTime = -1;
  }
}



#include "overhead_test.def.h"
