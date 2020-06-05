#ifndef _CKTRACKMESSAGES_H
#define _CKTRACKMESSAGES_H

#include "conv-header.h"
extern bool trackMessages;


#if CMK_ERROR_CHECKING
#include <unordered_map>
#include <vector>
#include <set>

#define CMI_UNIQ_MSG_ID(msg)         ((CmiMsgHeaderBasic *)msg)->uniqMsgId
#define CMI_SRC_PE(msg)              ((CmiMsgHeaderBasic *)msg)->msgSrcPe
#define CMI_MSG_LAYER_TYPE(msg)      ((CmiMsgHeaderBasic *)msg)->msgLayerType
#define CMI_MSG_COMM_SENDER(msg)     ((CmiMsgHeaderBasic *)msg)->commSender
#define CMI_SRC_NODE(msg)            ((CmiMsgHeaderBasic *)msg)->msgSrcNode
#define CMI_NETWORK_MSG(msg)            ((CmiMsgHeaderBasic *)msg)->networkMsg

struct msgInfo {
  int msgHandler;
  int type;
  int ep; // -1 for conv, lrts
  std::vector<int> destPes;
  bool nodeLevel;
  bool networkMsg;

  bool leftSender;
};

typedef std::unordered_map<int, int> CmiIntIntMap;
typedef std::unordered_map<int, msgInfo> CmiIntMsgInfoMap;

typedef std::set<std::pair<int, int>> CmiIntIntPair;

struct trackingAckMsg {
  char core[CmiMsgHeaderSizeBytes];
  int senderUniqId;
  int senderPe;
};

typedef int (*charmLevelFn)(void *msg);


void CmiMessageTrackerCharmInit(charmLevelFn fn);
void CmiMessageTrackerInit();
int getNewUniqId();
void addToTracking(char *msg, int destPe, bool nodeLevel, bool networkMsg);
void sendTrackingAck(char *msg);
void _receiveTrackingAck(trackingAckMsg *ackMsg);

void setMsgLeftSender(char *msg);

void addToRecvedUnackedMsgs(char *msg);
#endif
#endif
