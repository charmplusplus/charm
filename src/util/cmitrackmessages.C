#include "converse.h"
#include "cmitrackmessages.h"
#include <algorithm>

#define DEBUG(x) //x

// boolean set if user passes +trackMsgs (used to determine if tracking is enabled)
bool trackMessages;

#if CMK_ERROR_CHECKING

charmLevelFn getMsgEpIdxFn;

// uniqMsgId
CpvDeclare(int, uniqMsgId);

// Declare a pe level datastructure that stores the ids of outgoing messages
CpvDeclare(CmiIntMsgInfoMap, sentUniqMsgIds);

// Converse handler for ack messages
CpvDeclare(int, msgTrackHandler);

// Converse handler for when stats are done
CpvDeclare(int, statsDoneHandler);

// Converse handler for broadcasting to all PEs to print their stats
CpvDeclare(int, printStatsHandler);

inline void markAcked(char *msg) {
  CMI_UNIQ_MSG_ID(msg) = -10;
}

// Converse handler to receive an ack message
void _receiveTrackingAck(trackingAckMsg *ackMsg) {

  // Update data structure removing the entry associated with the pe
  CmiIntMsgInfoMap::iterator iter;
  std::vector<int>::iterator iter2;
  int uniqId = ackMsg->senderUniqId;
  int dest = CMI_SRC_PE(ackMsg);
  iter = CpvAccess(sentUniqMsgIds).find(uniqId);
  msgInfo info;

  if(iter != CpvAccess(sentUniqMsgIds).end()) {

    info = iter->second;

    if(info.nodeLevel)
      dest = CmiNodeOf(dest);

    iter2 = find(info.destPes.begin(), info.destPes.end(), dest);

    if(iter2 != info.destPes.end()) {
      // element found, remove it
      info.destPes.erase(iter2);

    } else {
      CmiPrintf("[%d][%d][%d] Sender valid dest pes for msgId:%d are:\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
      for(int i=0; i<info.destPes.size(); i++) {
        CmiPrintf("[%d][%d][%d] Pe:%d = [%d]\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, info.destPes[i]);
      }
      CmiAbort("[%d][%d][%d] ******* Sender Invalid Pe:%d (other Pe:%d) returned back for msg id:%d and msg:%p and nodeLevel is %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), dest, ackMsg->senderPe, ackMsg->senderUniqId, ackMsg, info.nodeLevel);
    }

    if(info.destPes.size() == 0) { // last count, remove map entry
      DEBUG(CmiPrintf("[%d][%d][%d] ERASING (uniqId:%d, pe:%d, type:%d, count:%zu, msgHandler:%d, ep:%d), remaining unacked messages = %zu \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.destPes.size(), info.msgHandler, info.ep, CpvAccess(sentUniqMsgIds).size());)
      CpvAccess(sentUniqMsgIds).erase(iter);

    } else {
      DEBUG(CmiPrintf("[%d][%d][%d] DECREMENTING COUNTER (uniqId:%d, pe:%d, type:%d, count:%zu, msgHandler:%d, ep:%d), remaining unacked messages = %zu \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.destPes.size(), info.msgHandler, info.ep, CpvAccess(sentUniqMsgIds).size());)
    }
  } else {
    //CmiPrintf("[%d][%d][%d] Sender Invalid msg id:%d returned back\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId);
    CmiAbort("[%d][%d][%d] Sender Invalid msg id:%d returned back from PE:%d node:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId, CMI_SRC_PE(ackMsg), CMI_SRC_NODE(ackMsg));
  }
  CmiFree(ackMsg);
}

void doneAllStats(char *msg) {
  CmiPrintf("[%d][%d][%d] =============== Message tracking - All stats done, broadcast to everyone to print stats ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
  char *bcastMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(bcastMsg, CpvAccess(printStatsHandler));
  CMI_UNIQ_MSG_ID(bcastMsg) = -11;
  CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, bcastMsg);
}

void *reductionMergeFn(int *size, void *data, void **remote, int count) {
  CmiPrintf("[%d][%d][%d] =============== Message tracking - REDUCTION ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
  *size = CmiMsgHeaderSizeBytes;
  char *reduceMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(reduceMsg, CpvAccess(statsDoneHandler));
  return reduceMsg;
}

void printStats() {
  CmiIntMsgInfoMap::iterator iter = CpvAccess(sentUniqMsgIds).begin();

  if(CpvAccess(sentUniqMsgIds).size() == 0) {
    CmiPrintf("[%d][%d][%d] =============== Message tracking - NO ENTRIES PRESENT ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank());
  } else {

    CmiPrintf("[%d][%d][%d] =============== Message tracking Num entries=%zu ==============\n",CmiMyPe(), CmiMyNode(), CmiMyRank(), CpvAccess(sentUniqMsgIds).size());
    int i = 1;
    msgInfo info;

    while(iter != CpvAccess(sentUniqMsgIds).end()) {
      info = iter->second;
      CmiPrintf("[%d][%d][%d] Main Entry:%d, PRINTING uniqId:%d, pe:%d, type:%d, count:%zu, msgHandler:%d, ep:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, iter->first, CmiMyPe(), info.type, info.destPes.size(), info.msgHandler, info.ep);
      for(int j = 0; j < info.destPes.size(); j++) {
        CmiPrintf("[%d][%d][%d] Main Entry:%d Sub Entry:%d, PRINTING uniqId:%d, pe:%d, type:%d, msgHandler:%d, ep:%d, destPe %d = %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, j, iter->first, CmiMyPe(), info.type, info.msgHandler, info.ep, j, info.destPes[j]);
      }
      i++;
      iter++;
    }
  }

  CsdExitScheduler();
}

void _printStats(char *msg) {
  CmiFree(msg);
  printStats();
}

void CmiPrintMTStatsOnIdle() {
  CmiPrintf("[%d][%d][%d] CmiPrintMTStatsOnIdle: CmiProcessor is idle\n",CmiMyPe(), CmiMyNode(), CmiMyRank());

  char *reduceMsg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(reduceMsg, CpvAccess(statsDoneHandler));
  CMI_UNIQ_MSG_ID(reduceMsg) = -12;
  CmiReduce(reduceMsg, CmiMsgHeaderSizeBytes, reductionMergeFn);
}


void CmiMessageTrackerCharmInit(charmLevelFn fn) {
  getMsgEpIdxFn = fn;
}

void CmiMessageTrackerInit() {

  DEBUG(CmiPrintf("[%d][%d][%d] CmiMessageTrackerInit\n", CmiMyPe(), CmiMyNode(), CmiMyRank());)
  CpvInitialize(CmiIntMsgInfoMap, sentUniqMsgIds);

  CpvInitialize(int, uniqMsgId);
  CpvAccess(uniqMsgId) = 0;

  CpvInitialize(int, msgTrackHandler);
  CpvAccess(msgTrackHandler) = CmiRegisterHandler((CmiHandler) _receiveTrackingAck);

  CpvInitialize(int, statsDoneHandler);
  CpvAccess(statsDoneHandler) = CmiRegisterHandler((CmiHandler) doneAllStats);

  CpvInitialize(int, printStatsHandler);
  CpvAccess(printStatsHandler) = CmiRegisterHandler((CmiHandler) _printStats);

  CcdCallOnCondition(CcdPROCESSOR_LONG_IDLE, (CcdVoidFn)CmiPrintMTStatsOnIdle, NULL);
}

// Method will be used to set a new uniq id
// will be called from charm, converse, machine layers
inline int getNewUniqId() {
  return ++CpvAccess(uniqMsgId);
}

inline void insertUniqIdEntry(char *msg, int destPe, bool nodeLevel, bool networkMsg) {
  int uniqId = getNewUniqId();

  CMI_UNIQ_MSG_ID(msg) = uniqId;

#if CMK_SMP
  if(CmiMyRank() == CmiMyNodeSize()) {
    CMI_MSG_COMM_SENDER(msg) = 1;
    CMI_SRC_NODE(msg) = CmiMyNode();
  } else
#endif
  {
    CMI_MSG_COMM_SENDER(msg) = 0;
  }

  CMI_SRC_PE(msg) = CmiMyPe();
  msgInfo info;
  info.type = CMI_MSG_LAYER_TYPE(msg);
  info.destPes.push_back(destPe);

  info.msgHandler = CmiGetHandler(msg);

  if(info.type) { // it's a charm message
    info.ep = getMsgEpIdxFn(msg);
  } else {
    info.ep = 0;
  }

  info.nodeLevel = nodeLevel;
  info.networkMsg = networkMsg;

  DEBUG(CmiPrintf("[%d][%d][%d] ADDING uniqId:%d, pe:%d, type:%d, count:%zu, msgHandler:%d, ep:%d, destPe:%d, isNodeLevel:%d, isNetworkMsg:%d, msg:%p\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.destPes.size(), info.msgHandler, info.ep, destPe, nodeLevel, networkMsg, msg);)

  CpvAccess(sentUniqMsgIds).insert({uniqId, info});
}

void addToTracking(char *msg, int destPe, bool nodeLevel, bool networkMsg) {

  // Do not track ack messages
  if(CmiGetHandler(msg) == CpvAccess(msgTrackHandler) || CMI_UNIQ_MSG_ID(msg) == -14) {
    // Do nothing
    return;
  }

  int uniqId = CMI_UNIQ_MSG_ID(msg);

  if(uniqId == -11 || uniqId == -12 || uniqId == -13) {
    // do not track
    return;
  }

  if(uniqId <= 0) {
    // uniqId not yet set
    insertUniqIdEntry(msg, destPe, nodeLevel, networkMsg);
  } else {
    // uniqId already set, increase count
    CmiIntMsgInfoMap::iterator iter;
    iter = CpvAccess(sentUniqMsgIds).find(uniqId);

    if(iter != CpvAccess(sentUniqMsgIds).end()) {
      msgInfo info = iter->second;

      if(nodeLevel != info.nodeLevel) {
        // should add a new entry
        insertUniqIdEntry(msg, destPe, nodeLevel, networkMsg);

      } else {
        //iter->second.count++; // increment counter
        iter->second.destPes.push_back(destPe);
        DEBUG(CmiPrintf("[%d][%d][%d] INCREMENTING COUNTER uniqId:%d, pe:%d, type:%d, count:%zu, msgHandler:%d, ep:%d, destPe:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, CmiMyPe(), info.type, info.destPes.size(), info.msgHandler, info.ep, destPe);)
      }
    } else {
      insertUniqIdEntry(msg, destPe, nodeLevel, networkMsg);
    }
  }
}

// Called when the message has been received
// will be called from charm, converse, machine layers
void sendTrackingAck(char *msg) {

  if(msg == NULL) {
    CmiAbort("[%d][%d][%d] Receiver received message is NULL\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
  }

  int uniqId = CMI_UNIQ_MSG_ID(msg);

  if(uniqId <= 0) {

    //CmiPrintf("[%d][%d][%d] Receiver received message with invalid id:%d and msg is %p\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), uniqId, msg);
    CmiAbort("[%d][%d][%d] Receiver received message %p with invalid id:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), msg, uniqId);

  } else {

    trackingAckMsg *ackMsg = (trackingAckMsg *)CmiAlloc(sizeof(trackingAckMsg));
    ackMsg->senderUniqId = CMI_UNIQ_MSG_ID(msg);
    ackMsg->senderPe = CmiMyPe();

    int srcPe = CMI_SRC_PE(msg);

    CMI_SRC_PE(ackMsg)      = CmiMyPe();
    CMI_UNIQ_MSG_ID(ackMsg) = -14;


    if(CMI_MSG_NOKEEP(msg)) {
      // Same message reused, ensure that the count drops to 1
      if(CmiGetReference(msg) == 1) {
        // To deal with messages that get enqueued twice
        markAcked(msg);
      }
    } else {
      // To deal with messages that get enqueued twice
      markAcked(msg);
    }

    CmiSetHandler(ackMsg, CpvAccess(msgTrackHandler));

#if CMK_SMP
    if(CMI_MSG_COMM_SENDER(msg) == 1) {
      CmiBecomeImmediate(ackMsg); // make it IMMEDIATE so that it is received on the comm thread
      if(CMI_SRC_NODE(msg) == CmiMyNode()) {
        DEBUG(CmiPrintf("[%d][%d][%d] ACKING with uniqId:%d back to my comm thread rank:%d, CMI_SRC_PE(msg)=%d, CMI_SRC_PE(ackMsg)=%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId, CmiMyNodeSize(), CMI_SRC_PE(msg), CMI_SRC_PE(ackMsg));)
        CmiPushPE(CmiMyNodeSize(), ackMsg);
      } else {
        DEBUG(CmiPrintf("[%d][%d][%d] ACKING with uniqId:%d back to other comm thread node:%d, CMI_SRC_PE(msg)=%d, CMI_SRC_PE(ackMsg)=%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId, CMI_SRC_NODE(msg), CMI_SRC_PE(msg), CMI_SRC_PE(ackMsg));)
        CmiSyncNodeSendAndFree(CMI_SRC_NODE(msg), sizeof(trackingAckMsg), ackMsg);
      }
    } else
#endif
    {
      DEBUG(CmiPrintf("[%d][%d][%d] ACKING with uniqId:%d back to pe:%d, CMI_SRC_PE(msg)=%d, CMI_SRC_PE(ackMsg)=%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ackMsg->senderUniqId, srcPe, CMI_SRC_PE(msg), CMI_SRC_PE(ackMsg));)
      CmiSyncSendAndFree(CMI_SRC_PE(msg), sizeof(trackingAckMsg), ackMsg);
    }
  }
}

#endif
