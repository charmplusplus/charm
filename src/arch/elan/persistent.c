/** @file
 * Elan persistent communication
 * @ingroup Machine
*/

/*
  included in machine.c
  common code for persistent communication now is moved to persist-comm.c
  Gengbin Zheng, 12/5/2003
*/

/*
  machine specific persistent comm functions:
  * CmiSendPersistentMsg
  * CmiSyncSendPersistent
  * PumpPersistent
  * PerAlloc PerFree      // persistent message memory allocation/free functions
  * persist_machine_init  // machine specific initialization call
*/

#define STRATEGY_ONE_ELANPUT   0
#define STRATEGY_TWO_ELANPUT   1

/****************************************************************************
       Pending send messages
****************************************************************************/
typedef struct pmsg_list {
  ELAN_EVENT *e;
  char *msg;
  struct pmsg_list *next;
  int size, destpe;
  void *addr;
  PersistentHandle  h;
  int strategy, phase;
} PMSG_LIST;

static PMSG_LIST *pending_persistent_msgs = NULL;
static PMSG_LIST *end_pending_persistent_msgs = NULL;
static PMSG_LIST *free_list_head = NULL;

/* free_list_head keeps a list of reusable PMSG_LIST */
#define NEW_PMSG_LIST(evt, m, s, dest, _addr, ph, stra) \
  if (free_list_head) { msg_tmp = free_list_head; free_list_head=free_list_head->next; }  \
  else msg_tmp = (PMSG_LIST *) CmiAlloc(sizeof(PMSG_LIST));	\
  msg_tmp->msg = m;	\
  msg_tmp->e = evt;	\
  msg_tmp->size = s;	\
  msg_tmp->next = NULL;	\
  msg_tmp->destpe = dest;	\
  msg_tmp->addr = _addr;	\
  msg_tmp->h = ph;		\
  msg_tmp->phase = 0;	\
  msg_tmp->strategy = stra;	

#define APPEND_PMSG_LIST(msg_tmp)	\
  if (pending_persistent_msgs==0)	\
    pending_persistent_msgs = msg_tmp;	\
  else	\
    end_pending_persistent_msgs->next = msg_tmp;	\
  end_pending_persistent_msgs = msg_tmp;


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

/*CmiPrintf("[%d] CmiSendPersistentMsg h=%p hdl=%d destPE=%d destAddress=%p size=%d\n", CmiMyPe(), *phs, CmiGetHandler(m), destPE, slot->destAddress[0], size);*/

  if (slot->destAddress[0]) {
    ELAN_EVENT *e1, *e2;
    int strategy = STRATEGY_ONE_ELANPUT;
    /* if (size > 280) strategy = STRATEGY_TWO_ELANPUT; */
    int *footer = (int*)((char*)m + size);
    footer[0] = size;
    footer[1] = 1;
    if (strategy == STRATEGY_ONE_ELANPUT) CMI_MESSAGE_SIZE(m) = size;
    else CMI_MESSAGE_SIZE(m) = 0;
    e1 = elan_put(elan_base->state, m, slot->destAddress[0], size+sizeof(int)*2, destPE);
    switch (strategy ) {
    case STRATEGY_ONE_ELANPUT:
    case STRATEGY_TWO_ELANPUT:  {
      PMSG_LIST *msg_tmp;
      NEW_PMSG_LIST(e1, m, size, destPE, slot->destSizeAddress[0], h, strategy);
      APPEND_PMSG_LIST(msg_tmp);
      swapSendSlotBuffers(slot);
      break;
      }
    case 2:
      elan_wait(e1, ELAN_POLL_EVENT);
      e2 = elan_put(elan_base->state, &size, slot->destSizeAddress[0], sizeof(int), destPE);
      elan_wait(e2, ELAN_POLL_EVENT);
      CMI_MESSAGE_SIZE(m) = 0;
      /*CmiPrintf("[%d] elan finished. \n", CmiMyPe());*/
      CmiFree(m);
    }
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
    CmiPrintf("[%d]Slot sending message directly\n", CmiMyPe());
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

  /*  CmiPrintf("Setting root to %d\n", 0); */
  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiSendPersistentMsg(h, destPE, size, dupmsg);
}

/* 
  1: finish the first put but still need to be in the queue for the second put.
  2: finish and should be removed from queue.
*/
static int remote_put_done(PMSG_LIST *smsg)
{
  int flag = elan_poll(smsg->e, ELAN_POLL_EVENT);
  if (flag) {
      switch (smsg->strategy) {
      case 0: 
        smsg->phase = 1;
        CmiFree(smsg->msg);
        return 2;
      case 1:
        if (smsg->phase == 1) {
          /*
            CmiPrintf("remote_put_done on %d\n", CmiMyPe());
          */
          return 2;
        }
        else {
          smsg->phase = 1;
          CmiFree(smsg->msg);
                                                                                
          PersistentSendsTable *slot = (PersistentSendsTable *)(smsg->h);
          smsg->e = elan_put(elan_base->state, &smsg->size, smsg->addr, sizeof(int), smsg->destpe);
          return 1;
        }
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
      /*CmiFree(msg_tmp);*/
      if (free_list_head) { msg_tmp->next = free_list_head; free_list_head = msg_tmp; }
      else free_list_head = msg_tmp;
      msg_tmp = temp;
    }
    else {
      prev = msg_tmp;
      msg_tmp = msg_tmp->next;
    }
  }
  end_pending_persistent_msgs = prev;
}

extern void CmiReference(void *blk);

/* called in PumpMsgs */
int PumpPersistent()
{
  int status = 0;
  PersistentReceivesTable *slot = persistentReceivesTableHead;
  while (slot) {
    char *msg = slot->messagePtr[0];
    int size = *(slot->recvSizePtr[0]);
    if (size)
    {
      int *footer = (int*)(msg + size);
      if (footer[0] == size && footer[1] == 1) {
/*CmiPrintf("[%d] PumpPersistent messagePtr=%p size:%d\n", CmiMyPe(), slot->messagePtr, size);*/

#if 0
      void *dupmsg;
      dupmsg = CmiAlloc(size);
                                                                                
      _MEMCHECK(dupmsg);
      memcpy(dupmsg, msg, size);
      memset(msg, 0, size+2*sizeof(int));
      msg = dupmsg;
#else
      /* return messagePtr directly and user MUST make sure not to delete it. */
      /*CmiPrintf("[%d] %p size:%d rank:%d root:%d\n", CmiMyPe(), msg, size, CMI_DEST_RANK(msg), CMI_BROADCAST_ROOT(msg));*/

      CmiReference(msg);
      swapRecvSlotBuffers(slot);
#endif

      CmiPushPE(CMI_DEST_RANK(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT(msg))
          SendSpanningChildren(size, msg);
#endif
      /* clear footer after message used */
      *(slot->recvSizePtr[0]) = 0;
      footer[0] = footer[1] = 0;

#if 0
      /* not safe at all! */
      /* instead of clear before use, do it earlier */
      msg=slot->messagePtr[0];
      size = *(slot->recvSizePtr[0]);
      footer = (int*)(msg + size);
      *(slot->recvSizePtr[0]) = 0;
      footer[0] = footer[1] = 0;
#endif
      status = 1;
      }
    }
    slot = slot->next;
  }
  return status;
}

void *PerAlloc(int size)
{
  return CmiAlloc(size);
}
                                                                                
void PerFree(char *msg)
{
  elan_CmiStaticFree(msg);
}

/* machine dependent init call */
void persist_machine_init(void)
{
}

void setupRecvSlot(PersistentReceivesTable *slot, int maxBytes)
{
  int i;
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
    char *buf = PerAlloc(maxBytes+sizeof(int)*2);
    _MEMCHECK(buf);
    memset(buf, 0, maxBytes+sizeof(int)*2);
    slot->messagePtr[i] = buf;
    /* note: assume first integer in elan converse header is the msg size */
    slot->recvSizePtr[i] = (unsigned int*)buf;
  }
  slot->sizeMax = maxBytes;
}


