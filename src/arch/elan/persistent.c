
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

/* Converse handler */
int persistentRequestHandlerIdx;
int persistenceReqGrantedHandlerIdx;

#define RESET 0
#define SET 1

typedef struct pmsg_list {
  ELAN_EVENT *e;
  char *msg;
  struct pmsg_list *next;
  int size, destpe;
  PersistentHandle  h;
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
  msg_tmp->h = ph;

#define APPEND_PMSG_LIST(msg_tmp)	\
  if (pending_persistent_msgs==0)	\
    pending_persistent_msgs = msg_tmp;	\
  else	\
    end_pending_persistent_msgs->next = msg_tmp;	\
  end_pending_persistent_msgs = msg_tmp;

int getFreeSendSlot()
{
  if (persistentSendsTableCount == TABLESIZE) CmiAbort("persistentSendsTable full.\n");
  return ++persistentSendsTableCount;
}

int getFreeRecvSlot()
{
  if (persistentReceivesTableCount == TABLESIZE) CmiAbort("persistentReceivesTable full.\n");
  return ++persistentReceivesTableCount;
}

PersistentHandle CmiCreatePersistent(int destPE, int maxBytes)
{
  PersistentSendsTable *slot;
  PersistentHandle h = getFreeSendSlot();
  if (h >= TABLESIZE) CmiAbort("persistentSendsTable full.\n");

  slot = &persistentSendsTable[h];
  slot->destPE = destPE;
  slot->sizeMax = maxBytes;
  slot->destAddress = NULL;
  slot->destSlotFlagAddress = NULL;
  slot->messagePtr = NULL;
  slot->flag = SET;

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
  PersistentSendsTable *slot;
  PersistentReqGrantedMsg *msg = (PersistentReqGrantedMsg *)env;
  int h = msg->sourceHandlerIndex;
  slot = &persistentSendsTable[h];
  slot->destSlotFlagAddress = msg->slotFlagAddress;
  slot->destAddress = msg->msgAddr;

  if (slot->messagePtr) {
    CmiSendPersistentMsg(h, slot->destPE, slot->messageSize, slot->messagePtr);
    slot->messagePtr = NULL;
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
#if 1
    PMSG_LIST *msg_tmp;
    NEW_PMSG_LIST(e1, m, size, destPE, h);
    APPEND_PMSG_LIST(msg_tmp);
#else
    elan_wait(e1, ELAN_POLL_EVENT);
    e2 = elan_put(elan_base->state, &size, slot->destSlotFlagAddress, sizeof(int), destPE);
    elan_wait(e2, ELAN_POLL_EVENT);
//CmiPrintf("[%d] elan finished. \n", CmiMyPe());
    CmiFree(m);
#endif
  }
  else {
#if 1
    if (slot->messagePtr != NULL) {
      CmiPrintf("Unexpected message in buffer on %d\n", CmiMyPe());
      CmiAbort("");
    }
    slot->messagePtr = m;
    slot->messageSize = size;
#else
  /* normal send */
  CmiSyncSendAndFree(slot->destPE, size, m);
#endif
  }
}

static int remote_put_done(PMSG_LIST *smsg)
{
  int flag = elan_poll(smsg->e, ELAN_POLL_EVENT);
  if (flag) {
    PersistentSendsTable *slot = &persistentSendsTable[smsg->h];
    ELAN_EVENT *e2 = elan_put(elan_base->state, &smsg->size, slot->destSlotFlagAddress, sizeof(int), smsg->destpe);
    elan_wait(e2, ELAN_POLL_EVENT);
    CmiFree(smsg->msg);
/*
CmiPrintf("remote_put_done on %d\n", CmiMyPe());
*/
  }
  return flag;
}

/* called in CmiReleaseSentMessages */
void release_pmsg_list()
{
  PMSG_LIST *prev=0, *temp;
  PMSG_LIST *msg_tmp = pending_persistent_msgs;

  while (msg_tmp) {
    if (remote_put_done(msg_tmp)) {
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

