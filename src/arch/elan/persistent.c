
#include "converse.h"

typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  void *destAddress;
  void *destSlotFlagAddress;
  void *messagePtr;
  int messageSize;
  int flag;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
  int flag;
  int sizeMax;
  void *messagePtr;
  int messageSize;
} PersistentReceivesTable;

#define TABLESIZE 1024
PersistentSendsTable persistentSendsTable[TABLESIZE];
int persistentSendsTableCount = 0;
PersistentReceivesTable persistentReceivesTable[TABLESIZE];
int persistentReceivesTableCount = 0;

typedef struct _PersistentRequestMsg {
  char core[CmiMsgHeaderSizeBytes];
  int maxBytes;
  PersistentHandle sourceHandlerIndex;
  int requestorPE;
} PersistentRequestMsg;

typedef struct _PersistentReqGrantedMsg {
  char core[CmiMsgHeaderSizeBytes];
  void *slotFlagAddress;
  void *msgAddr;
  PersistentHandle sourceHandlerIndex;
} PersistentReqGrantedMsg;

int persistentRequestHandlerIdx;
int persistenceReqGrantedHandlerIdx;

#define RESET 0
#define SET 1

int getFreeRecvSlot()
{
  if (persistentReceivesTableCount == TABLESIZE) CmiAbort("persistentReceivesTable full.\n");
  return ++persistentReceivesTableCount;
}

PersistentHandle CmiCreatePersistent(int destPE, int maxBytes)
{
  int h = ++persistentSendsTableCount; 
  if (h >= TABLESIZE) CmiAbort("persistentSendsTable full.\n");

  persistentSendsTable[h].destPE = destPE;
  persistentSendsTable[h].sizeMax = maxBytes;
  persistentSendsTable[h].destAddress = NULL;
  persistentSendsTable[h].destSlotFlagAddress = NULL;
  persistentSendsTable[h].messagePtr = NULL;
  persistentSendsTable[h].flag = SET;

  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandlerIndex = h;
  msg->requestorPE = CmiMyPe();

  CmiSetHandler(msg, persistentRequestHandlerIdx);
  CmiSyncSendAndFree(destPE,sizeof(PersistentRequestMsg),msg);

  return h;
}


void persistentRequestHandler(void *env)
{             
  PersistentRequestMsg *msg = (PersistentRequestMsg *)env;
  int slotIdx;
  PersistentReceivesTable *slot;

  slotIdx = getFreeRecvSlot();
  slot = &persistentReceivesTable[slotIdx];
  slot->messagePtr = CmiAlloc(msg->maxBytes);
  slot->flag = RESET;
  slot->sizeMax = msg->maxBytes;

  PersistentReqGrantedMsg *gmsg = CmiAlloc(sizeof(PersistentReqGrantedMsg));
  gmsg->slotFlagAddress = &slot->flag;
  gmsg->msgAddr = slot->messagePtr;
  gmsg->sourceHandlerIndex = msg->sourceHandlerIndex;

  CmiSetHandler(gmsg, persistenceReqGrantedHandlerIdx);
  CmiSyncSendAndFree(msg->requestorPE,sizeof(PersistentReqGrantedMsg),gmsg);
}

void persistenceReqGrantedHandler(void *env)
{
  PersistentReqGrantedMsg *msg = (PersistentReqGrantedMsg *)env;
  int h = msg->sourceHandlerIndex;
  persistentSendsTable[h].destSlotFlagAddress = msg->slotFlagAddress;
  persistentSendsTable[h].destAddress = msg->msgAddr;

  if (persistentSendsTable[h].messagePtr) {
    CmiSendPersistentMsg(h, persistentSendsTable[h].destPE, persistentSendsTable[h].messageSize, persistentSendsTable[h].messagePtr);
/*
    CmiSyncSend(persistentSendsTable[h].destPE, persistentSendsTable[h].messageSize, persistentSendsTable[h].messagePtr);
*/
    persistentSendsTable[h].messagePtr = NULL;
  }
}

void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m)
{
  PersistentSendsTable *slot = &persistentSendsTable[h];
  if (size > slot->sizeMax) {
    CmiPrintf("size: %d sizeMax: %d\n", size, slot->sizeMax);
    CmiAbort("Invalid size\n");
  }

/*
CmiPrintf("[%d] CmiSendPersistentMsg handle: %d\n", CmiMyPe(), h);
*/

  if (slot->destAddress) {
    ELAN_EVENT *e1, *e2;
    e1 = elan_put(elan_base->state, m, slot->destAddress, size, destPE);
    elan_wait(e1, ELAN_POLL_EVENT);
    e2 = elan_put(elan_base->state, &size, slot->destSlotFlagAddress, sizeof(int), destPE);
    elan_wait(e2, ELAN_POLL_EVENT);
/*
CmiPrintf("[%d] elan finished. \n", CmiMyPe());
*/
    CmiFree(m);
  }
  else {
    if (slot->messagePtr != NULL) {
      CmiPrintf("Unexpected message in buffer on %d\n", CmiMyPe());
      CmiAbort("");
    }
    slot->messagePtr = m;
    slot->messageSize = size;
  }
}

void PumpPersistent()
{
  int i;
  for (i=1; i<= persistentReceivesTableCount; i++) {
    if (persistentReceivesTable[i].flag != RESET) {
/*
CmiPrintf("PumpPersistent.\n");
*/
      int size = persistentReceivesTable[i].flag;
      void *msg = persistentReceivesTable[i].messagePtr;

#if 1
      // should return messagePtr directly and make sure keep it.
      void *dupmsg = CmiAlloc(size);
      memcpy(dupmsg, msg, size);
      msg = dupmsg;
#endif

      CmiPushPE(CMI_DEST_RANK(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
        if (CMI_BROADCAST_ROOT(msg))
          SendSpanningChildren(size, msg);
#endif
      persistentReceivesTable[i].flag = RESET;
    }
  }
}

void CmiPersistentInit()
{
  int i;
  persistentRequestHandlerIdx = CmiRegisterHandler((CmiHandler)persistentRequestHandler);
  persistenceReqGrantedHandlerIdx = CmiRegisterHandler((CmiHandler)persistenceReqGrantedHandler);
//CmiPrintf("CmiPersistentInit: %d %d \n", persistentRequestHandlerIdx, persistenceReqGrantedHandlerIdx);
  for (i=0; i<TABLESIZE; i++) {
    persistentSendsTable[i].destAddress = NULL;
    persistentSendsTable[i].destSlotFlagAddress = NULL;
    persistentSendsTable[i].messagePtr = NULL;
    persistentSendsTable[i].flag = RESET;
  }
  persistentSendsTableCount = 0;
  for (i=0; i<TABLESIZE; i++) {
    persistentReceivesTable[i].flag = RESET;
    persistentReceivesTable[i].messagePtr = NULL;
  }
  persistentReceivesTableCount = 0;
}


void CmiDestoryAllPersistent()
{
  int i;
  for (i=0; i<TABLESIZE; i++) {
    persistentSendsTable[i].destAddress = NULL;
    persistentSendsTable[i].destSlotFlagAddress = NULL;
    persistentSendsTable[i].messagePtr = NULL;
    persistentSendsTable[i].flag = RESET;
  }
  persistentSendsTableCount = 0;
  for (i=0; i<persistentReceivesTableCount; i++) {
    persistentReceivesTable[i].flag = RESET;
    if (persistentReceivesTable[i].messagePtr)
      CmiFree(persistentReceivesTable[i].messagePtr);
    persistentReceivesTable[i].messagePtr = NULL;
    persistentReceivesTable[i].messageSize = 0;
  }
  persistentReceivesTableCount = 0;
}


void CmiUsePersistentHandle(PersistentHandle *p, int n)
{
  phs = p;
  phsSize = n;
}

