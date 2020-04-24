#ifndef _CKTRACKMESSAGES_H
#define _CKTRACKMESSAGES_H

#include "conv-header.h"
extern bool trackMessages;


#if CMK_ERROR_CHECKING
#include <unordered_map>
#include <vector>

#define CMI_UNIQ_MSG_ID(msg)         ((CmiMsgHeaderBasic *)msg)->uniqMsgId
#define CMI_SRC_PE(msg)              ((CmiMsgHeaderBasic *)msg)->msgSrcPe
#define CMI_MSG_LAYER_TYPE(msg)      ((CmiMsgHeaderBasic *)msg)->msgLayerType

struct msgInfo {
  int msgHandler;
  int type;
  int ep; // -1 for conv, lrts
  std::vector<int> destPes;
};

typedef std::unordered_map<int, int> CmiIntIntMap;
typedef std::unordered_map<int, msgInfo> CmiIntMsgInfoMap;

struct trackingAckMsg {
  char core[CmiMsgHeaderSizeBytes];
  int senderUniqId;
};

typedef int (*charmLevelFn)(void *msg);


void CmiMessageTrackerInit(charmLevelFn fn);
int getNewUniqId();
void addToTracking(char *msg, int destPe);
void sendTrackingAck(char *msg);
void _receiveTrackingAck(trackingAckMsg *ackMsg);

#endif
#endif
