
#include "converse.h"

typedef struct _PersistentSendsTable {
  int destPE;
  int sizeMax;
  PersistentHandle   destHandle;  
  void *destAddress;
  void *destSizeAddress;
  void *messageBuf;
  int messageSize;
  char used;
} PersistentSendsTable;

typedef struct _PersistentReceivesTable {
  void *messagePtr;        /* preallocated message buffer of size "sizeMax" */
  int recvSize;
  int sizeMax;
  struct _PersistentReceivesTable *prev, *next;
} PersistentReceivesTable;

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

/****************************************************************************
       Pending send messages
****************************************************************************/
typedef struct pmsg_list {
  ELAN_EVENT *e;
  char *msg;
  struct pmsg_list *next;
  int size, destpe;
  PersistentHandle  h;
  int sent;
} PMSG_LIST;

static PMSG_LIST *pending_persistent_msgs = NULL;
static PMSG_LIST *end_pending_persistent_msgs = NULL;

#define NEW_PMSG_LIST(evt, m, s, dest, ph) \
  msg_tmp = (PMSG_LIST *) CmiAlloc(sizeof(PMSG_LIST));	\
  msg_tmp->msg = m;	\
  msg_tmp->e = evt;	\
  msg_tmp->size = s;	\
  msg_tmp->next = NULL;	\
  msg_tmp->destpe = dest;	\
  msg_tmp->h = ph;		\
  msg_tmp->sent = 0;

#define APPEND_PMSG_LIST(msg_tmp)	\
  if (pending_persistent_msgs==0)	\
    pending_persistent_msgs = msg_tmp;	\
  else	\
    end_pending_persistent_msgs->next = msg_tmp;	\
  end_pending_persistent_msgs = msg_tmp;

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
  slot->messagePtr = CmiAlloc(msg->maxBytes);
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

void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m)
{
  CmiAssert(h!=NULL);
  PersistentSendsTable *slot = (PersistentSendsTable *)h;
  CmiAssert(slot->used == 1);
  CmiAssert(slot->destPE == destPE);
  if (size > slot->sizeMax) {
    CmiPrintf("size: %d sizeMax: %d\n", size, slot->sizeMax);
    CmiAbort("Abort: Invalid size\n");
  }

//CmiPrintf("[%d] CmiSendPersistentMsg h=%p hdl=%d destAddress=%p size=%d\n", CmiMyPe(), *phs, CmiGetHandler(m), slot->destAddress, size);

  if (slot->destAddress) {
    ELAN_EVENT *e1, *e2;
    e1 = elan_put(elan_base->state, m, slot->destAddress, size, destPE);
#if 1
    PMSG_LIST *msg_tmp;
    NEW_PMSG_LIST(e1, m, size, destPE, h);
    APPEND_PMSG_LIST(msg_tmp);
#else
    elan_wait(e1, ELAN_POLL_EVENT);
    e2 = elan_put(elan_base->state, &size, slot->destSizeAddress, sizeof(int), destPE);
    elan_wait(e2, ELAN_POLL_EVENT);
//CmiPrintf("[%d] elan finished. \n", CmiMyPe());
    CmiFree(m);
#endif
  }
  else {
#if 1
    if (slot->messageBuf != NULL) {
      CmiPrintf("Unexpected message in buffer on %d\n", CmiMyPe());
      CmiAbort("");
    }
    slot->messageBuf = m;
    slot->messageSize = size;
#else
  /* normal send */
  PersistentHandle  *phs_tmp = phs;
  int phsSize_tmp = phsSize;
  phs = NULL; phsSize = 0;
  CmiSyncSendAndFree(slot->destPE, size, m);
  phs = phs_tmp; phsSize = phsSize_tmp;
#endif
  }
}

void CmiSyncSendPersistent(int destPE, int size, char *msg, PersistentHandle h)
{
  CmiState cs = CmiGetState();
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);

  //  CmiPrintf("Setting root to %d\n", 0);
  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiSendPersistentMsg(h, destPE, size, dupmsg);
}

// 1: finish the first put but still need to be in the queue for the second put.
// 2: finish and should be removed from queue.
static int remote_put_done(PMSG_LIST *smsg)
{
  int flag = elan_poll(smsg->e, ELAN_POLL_EVENT);
  if (flag) {
    if (smsg->sent == 1) {
/*
CmiPrintf("remote_put_done on %d\n", CmiMyPe());
*/
      return 2;
    }
    else {
      smsg->sent = 1;
      CmiFree(smsg->msg);
      PersistentSendsTable *slot = (PersistentSendsTable *)(smsg->h);
      smsg->e = elan_put(elan_base->state, &smsg->size, slot->destSizeAddress, sizeof(int), smsg->destpe);
      return 1;
    }
  }
  return 0;
}

/* called in CmiReleaseSentMessages */
void release_pmsg_list()
{
  PMSG_LIST *prev=0, *temp;
  PMSG_LIST *msg_tmp = pending_persistent_msgs;

  while (msg_tmp) {
    int status = remote_put_done(msg_tmp);
    if (status == 2) {
      temp = msg_tmp->next;
      if (prev==0)
        pending_persistent_msgs = temp;
      else
        prev->next = temp;
      CmiFree(msg_tmp);
      msg_tmp = temp;
    }
    else {
      prev = msg_tmp;
      msg_tmp = msg_tmp->next;
    }
  }
  end_pending_persistent_msgs = prev;
}

/* called in PumpMsgs */
void PumpPersistent()
{
  PersistentReceivesTable *slot = persistentReceivesTableHead;
  while (slot) {
    if (slot->recvSize)
    {
      int size = slot->recvSize;
      void *msg = slot->messagePtr;

#if 1
      // return messagePtr directly and user MUST make sure not to delete it.
      void *dupmsg = CmiAlloc(size);
      _MEMCHECK(dupmsg);
      memcpy(dupmsg, msg, size);
      msg = dupmsg;
#endif
//CmiPrintf("[%d] %p size:%d rank:%d root:%d\n", CmiMyPe(), msg, size, CMI_DEST_RANK(msg), CMI_BROADCAST_ROOT(msg));

      CmiPushPE(CMI_DEST_RANK(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
        if (CMI_BROADCAST_ROOT(msg))
          SendSpanningChildren(size, msg);
#endif
      slot->recvSize = 0;
    }
    slot = slot->next;
  }
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

  if (slot->messagePtr) CmiFree(slot->messagePtr);
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
    if (slot->messagePtr) CmiFree(slot->messagePtr);
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

