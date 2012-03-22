/** @file
 * Support for persistent communication setup
 * @ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

#include "converse.h"

#if CMK_PERSISTENT_COMM

#include "machine-persistent.h"

CpvDeclare(int, TABLESIZE);

CpvDeclare(PersistentSendsTable *, persistentSendsTable);
CpvDeclare(int, persistentSendsTableCount);
CpvDeclare(PersistentReceivesTable *, persistentReceivesTableHead);
CpvDeclare(PersistentReceivesTable *, persistentReceivesTableTail);
CpvDeclare(int, persistentReceivesTableCount);

/* Converse message type */
typedef struct _PersistentRequestMsg {
  char core[CmiMsgHeaderSizeBytes];
  int requestorPE;
  int maxBytes;
  PersistentHandle sourceHandler;
} PersistentRequestMsg;

typedef struct _PersistentReqGrantedMsg {
  char core[CmiMsgHeaderSizeBytes];
/*
  void *msgAddr[PERSIST_BUFFERS_NUM];
  void *slotFlagAddress[PERSIST_BUFFERS_NUM];
*/
  PersistentBuf    buf[PERSIST_BUFFERS_NUM];
  PersistentHandle sourceHandler;
  PersistentHandle destHandler;
} PersistentReqGrantedMsg;

typedef struct _PersistentDestoryMsg {
  char core[CmiMsgHeaderSizeBytes];
  PersistentHandle destHandlerIndex;
} PersistentDestoryMsg;

/* Converse handler */
int persistentRequestHandlerIdx;
int persistentReqGrantedHandlerIdx;
int persistentDestoryHandlerIdx;

CpvDeclare(PersistentHandle *, phs);
CpvDeclare(int, phsSize);
CpvDeclare(int, curphs);

/******************************************************************************
     Utilities
******************************************************************************/

extern void initRecvSlot(PersistentReceivesTable *slot);
extern void initSendSlot(PersistentSendsTable *slot);

void swapSendSlotBuffers(PersistentSendsTable *slot)
{
  if (PERSIST_BUFFERS_NUM == 2) {
#if 0
  void *tmp = slot->destAddress[0];
  slot->destAddress[0] = slot->destAddress[1];
  slot->destAddress[1] = tmp;
  tmp = slot->destSizeAddress[0];
  slot->destSizeAddress[0] = slot->destSizeAddress[1];
  slot->destSizeAddress[1] = tmp;
#else
  PersistentBuf tmp = slot->destBuf[0];
  slot->destBuf[0] = slot->destBuf[1];
  slot->destBuf[1] = tmp;
#endif
  }
}

void swapRecvSlotBuffers(PersistentReceivesTable *slot)
{
  if (PERSIST_BUFFERS_NUM == 2) {
#if 0
  void *tmp = slot->messagePtr[0];
  slot->messagePtr[0] = slot->messagePtr[1];
  slot->messagePtr[1] = tmp;
  tmp = slot->recvSizePtr[0];
  slot->recvSizePtr[0] = slot->recvSizePtr[1];
  slot->recvSizePtr[1] = tmp;
#else
  PersistentBuf tmp = slot->destBuf[0];
  slot->destBuf[0] = slot->destBuf[1];
  slot->destBuf[1] = tmp;
#endif
  }
}

PersistentHandle getFreeSendSlot()
{
  int i;
  if (CpvAccess(persistentSendsTableCount) == CpvAccess(TABLESIZE)) {
    CmiAbort("Charm++> too many persistent channels on sender.");
  }
  CpvAccess(persistentSendsTableCount)++;
  for (i=1; i<CpvAccess(TABLESIZE); i++)
    if (CpvAccess(persistentSendsTable)[i].used == 0) break;
  return &CpvAccess(persistentSendsTable)[i];
}

PersistentHandle getFreeRecvSlot()
{
  PersistentReceivesTable *slot = (PersistentReceivesTable *)CmiAlloc(sizeof(PersistentReceivesTable));
  initRecvSlot(slot);
  if (CpvAccess(persistentReceivesTableHead) == NULL) {
    CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = slot;
  }
  else {
    CpvAccess(persistentReceivesTableTail)->next = slot;
    slot->prev = CpvAccess(persistentReceivesTableTail);
    CpvAccess(persistentReceivesTableTail) = slot;
  }
  CpvAccess(persistentReceivesTableCount)++;
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
  PersistentHandle h;
  PersistentSendsTable *slot;

  if (CmiMyNode() == CmiNodeOf(destPE)) return NULL;

/*
  if (CmiMyPe() == destPE) {
    CmiPrintf("[%d] CmiCreatePersistent Error>  setting up persistent communication to the same processor is not allowed.\n", CmiMyPe());
    CmiAbort("CmiCreatePersistent");
  }
*/

  h = getFreeSendSlot();
  slot = (PersistentSendsTable *)h;

  slot->used = 1;
  slot->destPE = destPE;
  slot->sizeMax = maxBytes;

  PersistentRequestMsg *msg = (PersistentRequestMsg *)CmiAlloc(sizeof(PersistentRequestMsg));
  msg->maxBytes = maxBytes;
  msg->sourceHandler = h;
  msg->requestorPE = CmiMyPe();

  CmiSetHandler(msg, persistentRequestHandlerIdx);
  CmiSyncSendAndFree(destPE,sizeof(PersistentRequestMsg),msg);

  return h;
}

static void persistentRequestHandler(void *env)
{             
  PersistentRequestMsg *msg = (PersistentRequestMsg *)env;
  char *buf;
  int i;

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;
  /*slot->messagePtr = elan_CmiStaticAlloc(msg->maxBytes);*/

  /* build reply message */
  PersistentReqGrantedMsg *gmsg = CmiAlloc(sizeof(PersistentReqGrantedMsg));

  setupRecvSlot(slot, msg->maxBytes);

  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    gmsg->msgAddr[i] = slot->messagePtr[i];
    gmsg->slotFlagAddress[i] = slot->recvSizePtr[i];
#else
    gmsg->buf[i] = slot->destBuf[i];
#endif
  }

  gmsg->sourceHandler = msg->sourceHandler;
  gmsg->destHandler = getPersistentHandle(h, 1);

  CmiSetHandler(gmsg, persistentReqGrantedHandlerIdx);
  CmiSyncSendAndFree(msg->requestorPE,sizeof(PersistentReqGrantedMsg),gmsg);

  CmiFree(msg);
}

static void persistentReqGrantedHandler(void *env)
{
  int i;

  PersistentReqGrantedMsg *msg = (PersistentReqGrantedMsg *)env;
  PersistentHandle h = msg->sourceHandler;
  PersistentSendsTable *slot = (PersistentSendsTable *)h;

  /* CmiPrintf("[%d] Persistent handler granted  h:%p\n", CmiMyPe(), h); */

  CmiAssert(slot->used == 1);


  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    slot->destAddress[i] = msg->msgAddr[i];
    slot->destSizeAddress[i] = msg->slotFlagAddress[i];
#else
    slot->destBuf[i] = msg->buf[i];
#endif
  }
  slot->destHandle = msg->destHandler;

  if (slot->messageBuf) {
    LrtsSendPersistentMsg(h, CmiNodeOf(slot->destPE), slot->messageSize, slot->messageBuf);
    slot->messageBuf = NULL;
  }
  CmiFree(msg);
}

/*
  Another API:
  receiver initiate the persistent communication
*/
PersistentReq CmiCreateReceiverPersistent(int maxBytes)
{
  PersistentReq ret;
  int i;

  PersistentHandle h = getFreeRecvSlot();
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  setupRecvSlot(slot, maxBytes);

  ret.pe = CmiMyPe();
  ret.maxBytes = maxBytes;
  ret.myHand = h;
  ret.bufPtr = (void **)malloc(PERSIST_BUFFERS_NUM*sizeof(void*));
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
#if 0
    ret.messagePtr[i] = slot->messagePtr[i];
    ret.recvSizePtr[i] = slot->recvSizePtr[i];
#else
    ret.bufPtr[i] = malloc(sizeof(PersistentBuf));
    memcpy(&ret.bufPtr[i], &slot->destBuf[i], sizeof(PersistentBuf));
#endif
  }

  return ret;
}

PersistentHandle CmiRegisterReceivePersistent(PersistentReq recvHand)
{
  int i;
  PersistentHandle h = getFreeSendSlot();

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  slot->used = 1;
  slot->destPE = recvHand.pe;
  slot->sizeMax = recvHand.maxBytes;

#if 0
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
    slot->destAddress[i] = recvHand.messagePtr[i];
    slot->destSizeAddress[i] = recvHand.recvSizePtr[i];
  }
#else
  memcpy(slot->destBuf, recvHand.bufPtr, PERSIST_BUFFERS_NUM*sizeof(PersistentBuf));
#endif
  slot->destHandle = recvHand.myHand;
  return h;
}

/******************************************************************************
     destory Persistent Comm handler
******************************************************************************/

/* Converse Handler */
void persistentDestoryHandler(void *env)
{             
  int i;
  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)env;
  PersistentHandle h = getPersistentHandle(msg->destHandlerIndex, 0);
  CmiAssert(h!=NULL);
  CmiFree(msg);
  PersistentReceivesTable *slot = (PersistentReceivesTable *)h;

  CpvAccess(persistentReceivesTableCount) --;
  if (slot->prev) {
    slot->prev->next = slot->next;
  }
  else
    CpvAccess(persistentReceivesTableHead) = slot->next;
  if (slot->next) {
    slot->next->prev = slot->prev;
  }
  else
    CpvAccess(persistentReceivesTableTail) = slot->prev;

  for (i=0; i<PERSIST_BUFFERS_NUM; i++) 
    if (slot->destBuf[i].destAddress) /*elan_CmiStaticFree(slot->messagePtr);*/
      PerFree((char*)slot->destBuf[i].destAddress);

  clearRecvSlot(slot);

  CmiFree(slot);
}

/* FIXME: need to buffer until ReqGranted message come back? */
void CmiDestoryPersistent(PersistentHandle h)
{
  if (h == NULL) return;

  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  /* CmiAssert(slot->destHandle != 0); */

  PersistentDestoryMsg *msg = (PersistentDestoryMsg *)
                              CmiAlloc(sizeof(PersistentDestoryMsg));
  msg->destHandlerIndex = slot->destHandle;

  CmiSetHandler(msg, persistentDestoryHandlerIdx);
  CmiSyncSendAndFree(slot->destPE,sizeof(PersistentDestoryMsg),msg);

  /* free this slot */
  initSendSlot(slot);

  CpvAccess(persistentSendsTableCount) --;
}


void CmiDestoryAllPersistent()
{
  int i;
  for (i=0; i<CpvAccess(TABLESIZE); i++) {
    if (CpvAccess(persistentSendsTable)[i].messageBuf) 
      CmiPrintf("Warning: CmiDestoryAllPersistent destoried buffered unsend message.\n");
    initSendSlot(&CpvAccess(persistentSendsTable)[i]);
  }
  CpvAccess(persistentSendsTableCount) = 0;

  PersistentReceivesTable *slot = CpvAccess(persistentReceivesTableHead);
  while (slot) {
    PersistentReceivesTable *next = slot->next;
    int i;
    for (i=0; i<PERSIST_BUFFERS_NUM; i++)  {
      if (slot->destBuf[i].destSizeAddress)
        CmiPrintf("Warning: CmiDestoryAllPersistent destoried buffered undelivered message.\n");
      if (slot->destBuf[i].destAddress) PerFree((char*)slot->destBuf[i].destAddress);
    }
    CmiFree(slot);
    slot = next;
  }
  CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = NULL;
  CpvAccess(persistentReceivesTableCount) = 0;
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

 
  CpvInitialize(PersistentHandle*, phs);
  CpvAccess(phs) = NULL;
  CpvInitialize(int, phsSize);
  CpvInitialize(int, curphs);
  CpvAccess(curphs) = 0;

  persist_machine_init();

  CpvInitialize(int, TABLESIZE);
  CpvAccess(TABLESIZE) = 512;

  CpvInitialize(PersistentSendsTable *, persistentSendsTable);
  CpvAccess(persistentSendsTable) = (PersistentSendsTable *)malloc(CpvAccess(TABLESIZE) * sizeof(PersistentSendsTable));
  for (i=0; i<CpvAccess(TABLESIZE); i++) {
    initSendSlot(&CpvAccess(persistentSendsTable)[i]);
  }
  CpvInitialize(int, persistentSendsTableCount);
  CpvAccess(persistentSendsTableCount) = 0;

  CpvInitialize(PersistentReceivesTable *, persistentReceivesTableHead);
  CpvInitialize(PersistentReceivesTable *, persistentReceivesTableTail);
  CpvAccess(persistentReceivesTableHead) = CpvAccess(persistentReceivesTableTail) = NULL;
  CpvInitialize(int, persistentReceivesTableCount);
  CpvAccess(persistentReceivesTableCount) = 0;
}


void CmiUsePersistentHandle(PersistentHandle *p, int n)
{
  if (n==1 && *p == NULL) { p = NULL; n = 0; }
#if  CMK_ERROR_CHECKING && 0
  {
  int i;
  for (i=0; i<n; i++)
    if (p[i] == NULL) CmiAbort("CmiUsePersistentHandle: invalid PersistentHandle.\n");
  }
#endif
  CpvAccess(phs) = p;
  CpvAccess(phsSize) = n;
  CpvAccess(curphs) = 0;
}

#endif

/*@}*/
