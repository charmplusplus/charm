#include "defs.h"

/// Worker constructor
Worker::Worker(WorkerInitMsg *initMsg) {
  myPID = initMsg->myPID;
  totalNumWorkers = initMsg->totalNumWorkers;
  patternNum = initMsg->patternNum;
  msgCount = 0;
  iterNum = 0;
  msgNum = 0;

  dbPrintf("Worker %d constructed\n", myPID);

  // send the first message(s) of the simulation
  WorkMsg *wm;
  switch (patternNum) {
  case 0:  // ring
    if (myPID == 0) {
      wm = new WorkMsg(myPID, 0);
      msgCount++;
      POSE_invoke(recvMessage(wm), Worker, (totalNumWorkers > 1) ? 1 : 0, (POSE_TimeType)P0_MSG_TRANSIT_TIME);
    }
    break;
  case 1: {  // short communication burst, long message send, repeat
    POSE_srand(time(NULL));
    wm = new WorkMsg(myPID, 0);
    POSE_local_invoke(pattern1Iter(wm), 1);
    break;
  }
  case 2: {  // simultaneous ring with elapse
    wm = new WorkMsg(myPID, 0);
    msgCount++;
    POSE_invoke(recvMessage(wm), Worker, (myPID + 1) % totalNumWorkers, (POSE_TimeType)0);
    break;
  }
  case 3: {  // simultaneous ring without elapse
    wm = new WorkMsg(myPID, 0);
    msgCount++;
    POSE_invoke(recvMessage(wm), Worker, (myPID + 1) % totalNumWorkers, (POSE_TimeType)0);
    break;
  }
  default:
    CkPrintf("Invalid pattern number: %d\n", patternNum);
    CkAbort("");
    break;
  }
}

/// Send a message to a Worker
void Worker::sendMessage() {
  dbPrintf("[ID=%d][ovt=%lld] sendMessage msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);

  int destWorker;
  POSE_TimeType msgTransitTime;
  WorkMsg *wm;

  // set message parameters based on the pattern
  switch (patternNum) {
  case 0:  // ring
    destWorker = (myPID + 1) % totalNumWorkers;
    msgTransitTime = (POSE_TimeType)P0_MSG_TRANSIT_TIME;
    wm = new WorkMsg(myPID, msgCount);
    msgCount++;
    break;
  case 1:  // short communication burst, long message send, repeat
    //destWorker = POSE_rand() % totalNumWorkers;
    destWorker = myPID;
    msgTransitTime = (POSE_TimeType)(P1_BASE_MSG_TRANSIT_TIME + (POSE_rand() % P1_MSG_TRANSIT_TIME_RANGE));
    wm = new WorkMsg(myPID, msgCount);
    msgCount++;
    break;
  case 2:  // simultaneous ring with elapse
    destWorker = (myPID + 1) % totalNumWorkers;
    msgTransitTime = (POSE_TimeType)P2_MSG_TRANSIT_TIME;
    wm = new WorkMsg(myPID, msgCount);
    msgCount++;
    break;
  case 3: {  // simultaneous ring without elapse
    destWorker = (myPID + 1) % totalNumWorkers;
    msgTransitTime = (POSE_TimeType)P3_MSG_TRANSIT_TIME;
    wm = new WorkMsg(myPID, msgCount);
    msgCount++;
    break;
  }
  default:
    CkPrintf("Invalid pattern number: %d\n", patternNum);
    CkAbort("");
    break;
  }

  // send message
  POSE_invoke(recvMessage(wm), Worker, destWorker, msgTransitTime);
}

/// ENTRY: Receive a message from a Worker
void Worker::recvMessage(WorkMsg *wm) {
  dbPrintf("[ID=%d][ovt=%lld] recvMessage msgCount=%d iterNum=%d msgNum=%d wm:[srcWorkerPID=%d msgID=%d]\n", 
	   myPID, ovt, msgCount, iterNum, msgNum, wm->srcWorkerPID, wm->msgID);

  switch (patternNum) {
  case 0:  // ring
    elapse(P0_ELAPSE_TIME);
    if (((wm->msgID * totalNumWorkers) + wm->srcWorkerPID) < (P0_MESSAGES_TO_SEND - 1)) {
      sendMessage();
    }
    break;
  case 1:  // short communication burst, long message send, repeat
    // do nothing; everything is done on the sending side
    break;
  case 2:  // simultaneous ring with elapse
    if (((wm->msgID * totalNumWorkers) + wm->srcWorkerPID) < (P2_MESSAGES_TO_SEND - 1)) {
    elapse(P2_ELAPSE_TIME);
      sendMessage();
    }
    break;
  case 3: {  // simultaneous ring without elapse
    if (((wm->msgID * totalNumWorkers) + wm->srcWorkerPID) < (P3_MESSAGES_TO_SEND - 1)) {
      // manually set evt for DOP analysis
#ifndef CMK_TRACE_DISABLED
      if ((pose_config.stats) && (pose_config.dop)) {
        parent->dop_override_evt = ovt + (POSE_TimeType)P3_ELAPSE_TIME;
      }
#endif
      sendMessage();
    }
    break;
  }
  default:
    CkPrintf("Invalid pattern number: %d\n", patternNum);
    CkAbort("");
    break;
  }
}

/// ENTRY: Send a message for pattern 1
void Worker::pattern1Iter(WorkMsg *wm) {
  dbPrintf("[ID=%d][ovt=%lld] pattern1Iter msgCount=%d iterNum=%d msgNum=%d\n", myPID, ovt, msgCount, iterNum, msgNum);

  // For each iter, send P1_MESSAGES_PER_ITER messages and then wait a
  // while before starting the next iter
  if (iterNum < P1_ITERS) {
    if (msgNum < P1_MESSAGES_PER_ITER) {
      sendMessage();
      WorkMsg *wm = new WorkMsg(myPID, -((iterNum * P1_MESSAGES_PER_ITER) + msgNum));
      POSE_local_invoke(pattern1Iter(wm), P1_SHORT_DELAY);
      msgNum++;
    } else {
      WorkMsg *wm = new WorkMsg(myPID, -((iterNum * P1_MESSAGES_PER_ITER) + msgNum));
      POSE_local_invoke(pattern1Iter(wm), P1_LARGE_DELAY);
      msgNum = 0;
      iterNum++;
    }
  }
}

/// Termination function called at the end of the simulation
void Worker::terminus() {
  dbPrintf("[ID=%d] Simulation finished\n", myPID);
}
