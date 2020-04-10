#ifndef _CKTRACKMESSAGES_H
#define _CKTRACKMESSAGES_H

#include "conv-header.h"
extern bool trackMessages;


#if CMK_ERROR_CHECKING
#include <unordered_map>

#define CMI_UNIQ_MSG_ID(msg)         ((CmiMsgHeaderBasic *)msg)->uniqMsgId
#define CMI_SRC_PE(msg)              ((CmiMsgHeaderBasic *)msg)->srcPe

typedef std::unordered_map<int, int> CmiIntIntMap;

struct trackingAckMsg {
  char core[CmiMsgHeaderSizeBytes];
  int senderUniqId;
};


void CmiMessageTrackerInit();
int getNewUniqId();
void addToTracking(char *msg);
void sendTrackingAck(char *msg);
void _receiveTrackingAck(trackingAckMsg *ackMsg);

#endif
#endif
