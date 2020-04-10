#include "converse.h"
#include "cmitrackmessages.h"

#define DEBUG(x) //x

charmLevelFn getMsgEpIdxFn;

// boolean to see if it is required to track messages
bool trackMessages;

#if CMK_ERROR_CHECKING

// uniqMsgId
CpvDeclare(int, uniqMsgId);

// Declare a pe level datastructure that stores the ids of outgoing messages
CpvDeclare(CmiIntMsgInfoMap, sentUniqMsgIds);

// Converse handler for ack messages
CpvDeclare(int, msgTrackHandler);

// Converse handler to receive an ack message
void _receiveTrackingAck(trackingAckMsg *ackMsg) {

  // Update data structure removing the entry associated with the pe
  CmiIntMsgInfoMap::iterator iter;
  int uniqId = ackMsg->senderUniqId;
  iter = CpvAccess(sentUniqMsgIds).find(uniqId);
  msgInfo info;

  if(iter != CpvAccess(sentUniqMsgIds).end()) {

    info = iter->second;

    if(iter->second.count == 1) { // last count
      DEBUG(CmiPrintf("[%d][%d][%d] ERASING (uniqId:%d, pe:%d, type:%d, count:%d, msgHandler:%d, ep:%d), remaining unacked messages = %zu \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.count, info.msgHandler, info.ep, CpvAccess(sentUniqMsgIds).size());)

      CpvAccess(sentUniqMsgIds).erase(iter);

    } else {

      iter->second.count--;
      DEBUG(CmiPrintf("[%d][%d][%d] DECREMENTING COUNTER (uniqId:%d, pe:%d, type:%d, count:%d, msgHandler:%d, ep:%d), remaining unacked messages = %zu \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.count, info.msgHandler, info.ep, CpvAccess(sentUniqMsgIds).size());)
    }
  } else {
    //CmiPrintf("[%d][%d][%d] Sender Invalid msg id:%d returned back\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
    CmiAbort("[%d][%d][%d] Sender Invalid msg id:%d returned back\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
  }
  CmiFree(ackMsg);
}


void printStats() {
  CmiIntMsgInfoMap::iterator iter = CpvAccess(sentUniqMsgIds).begin();
  if(CpvAccess(sentUniqMsgIds).size() == 0) {
    CmiPrintf("[%d][%d][%d] =============== Message tracking - NO ENTRIES PRESENT ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
    return;
  }

  CmiPrintf("[%d][%d][%d] =============== Message tracking Num entries=%zu ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank(), CpvAccess(sentUniqMsgIds).size());
  int i = 1;
  msgInfo info;

  while(iter != CpvAccess(sentUniqMsgIds).end()) {
    info = iter->second;
    CmiPrintf("[%d][%d][%d] Entry:%d, PRINTING uniqId:%d, pe:%d, type:%d, count:%d, msgHandler:%d, ep:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i++, iter->first, CmiMyPe(), info.type, info.count, info.msgHandler, info.ep);
    iter++;
  }
}

void CmiPrintMTStatsOnIdle() {
  CmiPrintf("[%d][%d][%d] CmiPrintMTStatsOnIdle: CmiProcessor is idle, printing stats\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
  printStats();
}


void CmiMessageTrackerInit(charmLevelFn fn) {
  CpvInitialize(int, uniqMsgId);
  CpvAccess(uniqMsgId) = 0;

  CpvInitialize(int, msgTrackHandler);
  CpvAccess(msgTrackHandler) = CmiRegisterHandler((CmiHandler) _receiveTrackingAck);

  getMsgEpIdxFn = fn;

  //CcdCallOnCondition(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiPrintMTStatsOnIdle, NULL);
}

// Method will be used to set a new uniq id
// will be called from charm, converse, machine layers
inline int getNewUniqId() {
  return ++CpvAccess(uniqMsgId);
}

inline void insertUniqIdEntry(char *msg) {
  int uniqId = getNewUniqId();

  CMI_UNIQ_MSG_ID(msg) = uniqId;
  CMI_SRC_PE(msg)      = CmiMyPe();

  msgInfo info;
  info.type = CMI_MSG_LAYER_TYPE(msg);
  info.count = 1;

  info.msgHandler = CmiGetHandler(msg);

  if(info.type) { // it's a charm message
    info.ep = getMsgEpIdxFn(msg);
  } else {
    info.ep = 0;
  }

  DEBUG(CmiPrintf("[%d][%d][%d] ADDING uniqId:%d, pe:%d, type:%d, count:%d, msgHandler:%d, ep:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.count, info.msgHandler, info.ep);)
  CpvAccess(sentUniqMsgIds).insert({uniqId, info});
}

void addToTracking(char *msg) {

  // Do not track ack messages
  if(CmiGetHandler(msg) == CpvAccess(msgTrackHandler)) {
    CMI_UNIQ_MSG_ID(msg) = -2;
    CMI_SRC_PE(msg)      = CmiMyPe();
    return;
  }

  int uniqId = CMI_UNIQ_MSG_ID(msg);

  if(uniqId <= 0) {
    // uniqId not yet set
    insertUniqIdEntry(msg);
  } else {
    // uniqId already set, increase count
    CmiIntMsgInfoMap::iterator iter;
    iter = CpvAccess(sentUniqMsgIds).find(uniqId);

    if(iter != CpvAccess(sentUniqMsgIds).end()) {
      iter->second.count++; // increment counter

      msgInfo info = iter->second;

      DEBUG(CmiPrintf("[%d][%d][%d] INCREMENTING COUNTER uniqId:%d, pe:%d, type:%d, count:%d, msgHandler:%d, ep:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.count, info.msgHandler, info.ep);)
    } else {
      insertUniqIdEntry(msg);
    }
  }
}

inline void markAcked(char *msg) {
  CMI_UNIQ_MSG_ID(msg) = -10;
}

// Called when the message has been received
// will be called from charm, converse, machine layers
void sendTrackingAck(char *msg) {

  if(msg == NULL) {
    CmiAbort("[%d][%d][%d] Receiver received message is NULL\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
  }

  int uniqId = CMI_UNIQ_MSG_ID(msg);

  if(uniqId <= 0) {

    //CmiPrintf("[%d][%d][%d] Receiver received message with invalid id:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId);
    CmiAbort("[%d][%d][%d] Receiver received message with invalid id:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId);

  } else {

    trackingAckMsg *ackMsg = (trackingAckMsg *)CmiAlloc(sizeof(trackingAckMsg));
    ackMsg->senderUniqId = CMI_UNIQ_MSG_ID(msg);

    // To deal with messages that get enqueued twice
    markAcked(msg);

    CmiSetHandler(ackMsg, CpvAccess(msgTrackHandler));

    DEBUG(CmiPrintf("[%d][%d][%d] ACKING with uniqId:%d back to pe:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId, CMI_SRC_PE(msg));)

    CmiSyncSendAndFree(CMI_SRC_PE(msg), sizeof(trackingAckMsg), ackMsg);
  }
}

#endif
