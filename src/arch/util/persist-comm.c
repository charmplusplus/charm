
#include "converse.h"

#if CMK_PERSISTENT_COMM

#include "persist_impl.h"

#define TABLESIZE 1024
PersistentSendsTable persistentSendsTable[TABLESIZE];
int persistentSendsTableCount = 0;
PersistentReceivesTable *persistentReceivesTableHead;
PersistentReceivesTable *persistentReceivesTableTail;
int persistentReceivesTableCount = 0;

/* Converse message type */
typedef struct _PersistentRequestMsg {
  char core[CmiMsgHeaderSizeBytes];
  int requestorPE;
  int maxBytes;
  PersistentHandle sourceHandlerIndex;
} PersistentRequestMsg;

typedef struct _PersistentReqGrantedMsg {
  char core[CmiMsgHeaderSizeBytes];
  void *slotFlagAddress;
  void *msgAddr;
  PersistentHandle sourceHandlerIndex;
  PersistentHandle destHandlerIndex;
} PersistentReqGrantedMsg;

typedef struct _PersistentDestoryMsg {
  char core[CmiMsgHeaderSizeBytes];
  PersistentHandle destHandlerIndex;
} PersistentDestoryMsg;

/* Converse handler */
int persistentRequestHandlerIdx;
int persistentReqGrantedHandlerIdx;
int persistentDestoryHandlerIdx;

PersistentHandle  *phs = NULL;
int phsSize;

void PumpPersistent();
void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m);
void CmiSyncSendPersistent(int destPE, int size, char *msg,
                           PersistentHandle h);

/******************************************************************************
     Utilities
******************************************************************************/

void initSendSlot(PersistentSendsTable *slot)
{
  slot->used = 0;
  slot->destPE = -1;
  slot->sizeMax = 0;
  slot->destHandle = 0; 
  slot->destAddress = NULL;
  slot->destSizeAddress = NULL;
  slot->messageBuf = 0;
  slot->messageSize = 0;
}

void initRecvSlot(PersistentReceivesTable *slot)
{
  slot->recvSize = 0;
  slot->sizeMax = 0;
  slot->messagePtr = NULL;
  slot->prev = slot->next = NULL;
}

PersistentHandle getFreeSendSlot()
{
  int i;
  if (persistentSendsTableCount == TABLESIZE) CmiAbort("persistentSendsTable full.\n");
  persistentSendsTableCount++;
  for (i=1; i<TABLESIZE; i++)
    if (persistentSendsTable[i].used == 0) break;
  return &persistentSendsTable[i];
}

PersistentHandle getFreeRecvSlot()
{
  PersistentReceivesTable *slot = (PersistentReceivesTable *)CmiAlloc(sizeof(PersistentReceivesTable));
  initRecvSlot(slot);
  if (persistentReceivesTableHead == NULL) {
    persistentReceivesTableHead = persistentReceivesTableTail = slot;
  }
  else {
    persistentReceivesTableTail->next = slot;
    slot->prev = persistentReceivesTableTail;
    persistentReceivesTableTail = slot;
  }
  persistentReceivesTableCount++;
  return slot;
}

/******************************************************************************
     Create Persistent Comm handler
     When creating a persistent comm with destPE and maxSize
     1. allocate a free PersistentSendsTable entry, send a 
        PersistentRequestMsg message to destPE
        buffer persistent message before  Persistent Comm is setup;
     2. destPE execute Converse handler persistentRequestHandler() on the
        PersistentRequestMsg message:
        allocate a free PersistentReceivesTable entry;
        allocate a message buffer of size maxSize for the communication;
        Send back a PersistentReqGrantedMsg with message address, etc for
        elan_put;
     3. Converse handler persistentReqGrantedHandler() executed on
        sender for the PersistentReqGrantedMsg. setup finish, send buffered
        message.
******************************************************************************/

PersistentHandle CmiCreatePersistent(int destPE, int maxBytes)
{
  PersistentHandle h = getFreeSendSlot();

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  slot->used = 1;
  slot->destPE = destPE;
  slot->sizeMax = maxBytes;

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

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;
  /*slot->messagePtr = elan_CmiStaticAlloc(msg->maxBytes);*/

  slot->messagePtr = PerAlloc(msg->maxBytes);

  _MEMCHECK(slot->messagePtr);
  slot->sizeMax = msg->maxBytes;

  PersistentReqGrantedMsg *gmsg = CmiAlloc(sizeof(PersistentReqGrantedMsg));
  gmsg->slotFlagAddress = &(slot->recvSize);
  gmsg->msgAddr = slot->messagePtr;
  gmsg->sourceHandlerIndex = msg->sourceHandlerIndex;
  gmsg->destHandlerIndex = h;

  CmiSetHandler(gmsg, persistentReqGrantedHandlerIdx);
  CmiSyncSendAndFree(msg->requestorPE,sizeof(PersistentReqGrantedMsg),gmsg);

  CmiFree(msg);
}

void persistentReqGrantedHandler(void *env)
{

  /*CmiPrintf("Persistent handler granted\n");*/
  PersistentReqGrantedMsg *msg = (PersistentReqGrantedMsg *)env;
  PersistentHandle h = msg->sourceHandlerIndex;
  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  CmiAssert(slot->used == 1);
  slot->destAddress = msg->msgAddr;
  slot->destSizeAddress = msg->slotFlagAddress;
  slot->destHandle = msg->destHandlerIndex;

  if (slot->messageBuf) {
    CmiSendPersistentMsg(h, slot->destPE, slot->messageSize, slot->messageBuf);
    slot->messageBuf = NULL;
  }
  CmiFree(msg);
}

/******************************************************************************
     destory Persistent Comm handler
******************************************************************************/

/* Converse Handler */
void persistentDestoryHandler(void *env)
{             
  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)env;
  PersistentHandle h = msg->destHandlerIndex;
  CmiAssert(h!=NULL);
  CmiFree(msg);
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  persistentReceivesTableCount --;
  if (slot->prev) {
    slot->prev->next = slot->next;
  }
  else
   persistentReceivesTableHead = slot->next;
  if (slot->next) {
    slot->next->prev = slot->prev;
  }
  else
    persistentReceivesTableTail = slot->prev;

  if (slot->messagePtr) /*elan_CmiStaticFree(slot->messagePtr);*/
      CmiFree(slot->messagePtr);

  CmiFree(slot);
}

/* FIXME: need to buffer until ReqGranted message come back? */
void CmiDestoryPersistent(PersistentHandle h)
{
  if (h == 0) CmiAbort("CmiDestoryPersistent: not a valid PersistentHandle\n");

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  CmiAssert(slot->destHandle != 0);

  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)
                              CmiAlloc(sizeof(PersistentDestoryMsg));
  msg->destHandlerIndex = slot->destHandle;

  CmiSetHandler(msg, persistentDestoryHandlerIdx);
  CmiSyncSendAndFree(slot->destPE,sizeof(PersistentDestoryMsg),msg);

  /* free this slot */
  initSendSlot(slot);

  persistentSendsTableCount --;
}


void CmiDestoryAllPersistent()
{
  int i;
  for (i=0; i<TABLESIZE; i++) {
    if (persistentSendsTable[i].messageBuf) 
      CmiPrintf("Warning: CmiDestoryAllPersistent destoried buffered unsend message.\n");
    initSendSlot(&persistentSendsTable[i]);
  }
  persistentSendsTableCount = 0;

  PersistentReceivesTable *slot = persistentReceivesTableHead;
  while (slot) {
    PersistentReceivesTable *next = slot->next;
    if (slot->recvSize)
      CmiPrintf("Warning: CmiDestoryAllPersistent destoried buffered undelivered message.\n");
    if (slot->messagePtr) PerFree(slot->messagePtr);
    CmiFree(slot);
    slot = next;
  }
  persistentReceivesTableHead = persistentReceivesTableTail = NULL;
  persistentReceivesTableCount = 0;
}

void CmiPersistentInit()
{
  int i;
  persistentRequestHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentRequestHandler);
  persistentReqGrantedHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentReqGrantedHandler);
  persistentDestoryHandlerIdx = 
       CmiRegisterHandler((CmiHandler)persistentDestoryHandler);

  persist_machine_init();

  for (i=0; i<TABLESIZE; i++) {
    initSendSlot(&persistentSendsTable[i]);
  }
  persistentSendsTableCount = 0;
  persistentReceivesTableHead = persistentReceivesTableTail = NULL;
  persistentReceivesTableCount = 0;
}


void CmiUsePersistentHandle(PersistentHandle *p, int n)
{
#ifndef CMK_OPTIMIZE
  int i;
  for (i=0; i<n; i++)
    if (p[i] == NULL) CmiAbort("CmiUsePersistentHandle: invalid PersistentHandle.\n");
#endif
  phs = p;
  phsSize = n;
}

#endif
