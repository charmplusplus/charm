#include "defs.h"

/// Initialization message used for constructing a Worker
class WorkerInitMsg {
 public:
  int myPID;           // poser ID
  int totalNumWorkers; // number of worker objects in the simulation
  int patternNum;      // communication pattern to use

  WorkerInitMsg(int pid, int totalWorkers, int pattern): myPID(pid), totalNumWorkers(totalWorkers), patternNum(pattern) {}
  WorkerInitMsg& operator=(const WorkerInitMsg& obj) {
    eventMsg::operator=(obj);
    myPID = obj.myPID;
    totalNumWorkers = obj.totalNumWorkers;
    patternNum = obj.patternNum;
    return *this;
  }
};


/// Message passed between Worker posers
class WorkMsg {
 public:
  int srcWorkerPID; // poser ID of the source Worker
  int msgID;        // ID of the message

  WorkMsg(int workerPID, int id): srcWorkerPID(workerPID), msgID(id) {}
  WorkMsg& operator=(const WorkMsg& obj) {
    eventMsg::operator=(obj);
    srcWorkerPID = obj.srcWorkerPID;
    msgID = obj.msgID;
    return *this;
  }
};


/// Posers that send messages to each other
class Worker {
 public:
  int myPID;           // poser ID
  int totalNumWorkers; // number of worker objects in the simulation
  int patternNum;      // communication pattern to use
  int msgCount;        // current message count (total for whole simulation)
  int iterNum;         // current iteration number
  int msgNum;          // current message number (for current iteration)

  Worker() {}
  Worker(WorkerInitMsg *m); 
  ~Worker() {}
  Worker& operator=(const Worker& obj) {
    rep::operator=(obj);
    myPID = obj.myPID;
    totalNumWorkers = obj.totalNumWorkers;
    patternNum = obj.patternNum;
    msgCount = obj.msgCount;
    iterNum = obj.iterNum;
    msgNum = obj.msgNum;
    return *this;
  }
  void pup(PUP::er &p) { 
    chpt<state_Worker>::pup(p); 
    p|myPID; p|totalNumWorkers; p|patternNum; 
    p|msgCount; p|iterNum; p|msgNum;
  }
  void dump() {
    CkPrintf("Worker: myPID=%d totalNumWorkers=%d patternNum=%d msgCount=%d iterNum=%d msgNum=%d\n", 
	     myPID, totalNumWorkers, patternNum, msgCount, iterNum, msgNum);
    chpt<state_Worker>::dump();
  }
  void terminus();

  /// Send a message to a Worker
  void sendMessage();

  /// ENTRY: Receive a message from a Worker
  void recvMessage(WorkMsg *wm);
  void recvMessage_anti(WorkMsg *wm) {
    dbPrintf("[ID=%d][ovt=%lld] recvMessage_anti msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);
    restore(this);
    dbPrintf("   ovt after rollback=%lld\n", ovt);
  }
  void recvMessage_commit(WorkMsg *wm) {
    dbPrintf("[ID=%d][ovt=%lld] recvMessage_commit msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);
  }

  /// ENTRY: Send a message for pattern 1
  void pattern1Iter(WorkMsg *wm);
  void pattern1Iter_anti(WorkMsg *wm) {
    dbPrintf("[ID=%d][ovt=%lld] pattern1Iter_anti msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);
    restore(this);
    dbPrintf("   ovt after rollback=%lld\n", ovt);
  }
  void pattern1Iter_commit(WorkMsg *wm) {
    dbPrintf("[ID=%d][ovt=%lld] pattern1Iter_commit msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);
  }
};
