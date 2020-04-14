#include "converse.h"
#include "cmitrackmessages.h"

#define DEBUG(x) x

// boolean to see if it is required to track messages
bool trackMessages;

#if CMK_ERROR_CHECKING

// uniqMsgId
CpvDeclare(int, uniqMsgId);

// Declare a pe level datastructure that stores the ids of outgoing messages
CpvDeclare(CmiIntIntMap, sentUniqMsgIds);

// Converse handler for ack messages
CpvDeclare(int, msgTrackHandler);

// Converse handler to receive an ack message
void _receiveTrackingAck(trackingAckMsg *ackMsg) {

  // Update data structure removing the entry associated with the pe
  std::unordered_map<int, int>::iterator iter;
  int uniqId = ackMsg->senderUniqId;
  iter = CpvAccess(sentUniqMsgIds).find(uniqId);

  if(iter != CpvAccess(sentUniqMsgIds).end()) {
    if(iter->second == 1) { // last count
      CpvAccess(sentUniqMsgIds).erase(iter);
      DEBUG(CmiPrintf("[%d][%d][%d] Erased ack with id %d, remaining unacked messages = %zu\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CpvAccess(sentUniqMsgIds).size());)
    } else {
      iter->second--;
      DEBUG(CmiPrintf("[%d][%d][%d] Decremented id %d remaining unacked messages = %zu\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CpvAccess(sentUniqMsgIds).size());)
    }
  } else {
    //CmiPrintf("[%d][%d][%d] Sender Invalid msg id:%d returned back\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
    CmiAbort("[%d][%d][%d] Sender Invalid msg id:%d returned back\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
  }
  CmiFree(ackMsg);
}


void printStats() {
  std::unordered_map<int, int>::iterator iter = CpvAccess(sentUniqMsgIds).begin();
  if(CpvAccess(sentUniqMsgIds).size() == 0) {
    CmiPrintf("[%d][%d][%d] =============== Message tracking - NO ENTRIES PRESENT ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
    return;
  }

  CmiPrintf("[%d][%d][%d] =============== Message tracking Num entries=%zu ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank(), CpvAccess(sentUniqMsgIds).size());
  int i = 1;
  while(iter != CpvAccess(sentUniqMsgIds).end()) {
    CmiPrintf("[%d][%d][%d] =============== Entry %d: UniqId(%d) => Count(%d) ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank(), i++, iter->first, iter->second);
    iter++;
  }
}

void CmiPrintMTStatsOnIdle() {
  CmiPrintf("[%d][%d][%d] CmiPrintMTStatsOnIdle: CmiProcessor is idle, printing stats\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
  printStats();
}


void CmiMessageTrackerInit() {
  CpvInitialize(int, uniqMsgId);
  CpvAccess(uniqMsgId) = 0;

  CpvInitialize(int, msgTrackHandler);
  CpvAccess(msgTrackHandler) = CmiRegisterHandler((CmiHandler) _receiveTrackingAck);

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
  DEBUG(CmiPrintf("[%d][%d][%d] Add new unique Id %d with count 1 \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId);)
  CpvAccess(sentUniqMsgIds).insert({uniqId, 1});
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
    std::unordered_map<int, int>::iterator iter;
    iter = CpvAccess(sentUniqMsgIds).find(uniqId);

    if(iter != CpvAccess(sentUniqMsgIds).end()) {
      iter->second++; // incremeent counter
      DEBUG(CmiPrintf("[%d][%d][%d] Update unique Id %d with count %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, iter->second);)
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
    CmiSyncSendAndFree(CMI_SRC_PE(msg), sizeof(trackingAckMsg), ackMsg);
  }
}

#endif
