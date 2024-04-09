/*
 * =====================================================================================
 *
 *       Filename:  machine-persistent.C
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/11/2013 14:33:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Yanhua), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "persist_impl.h"

#define CMI_DEST_RANK_NET(msg)	((CmiMsgHeaderBasic*)msg)->rank
int persistentSendMsgHandlerIdx;

static void sendPerMsgHandler(char *msg)
{
  int msgSize;
  void *destAddr, *destSizeAddr;
  int ep;

  msgSize = CMI_MSG_SIZE(msg);
  msgSize -= (2*sizeof(void *)+sizeof(int));
  ep = *(int*)(msg+msgSize);
  destAddr = *(void **)(msg + msgSize + sizeof(int));
  destSizeAddr = *(void **)(msg + msgSize + sizeof(int) + sizeof(void*));
/*CmiPrintf("msgSize:%d destAddr:%p, destSizeAddr:%p\n", msgSize, destAddr, destSizeAddr);*/
  CmiSetHandler(msg, ep);
  *((int *)destSizeAddr) = msgSize;
  memcpy(destAddr, msg, msgSize);
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

/*CmiPrintf("[%d] CmiSendPersistentMsg h=%p hdl=%d destpe=%d destAddress=%p size=%d\n", CmiMyPe(), *phs, CmiGetHandler(m), slot->destPE, slot->destAddress, size);*/

  if (slot->destAddress[0]) {
    int oldep = CmiGetHandler(m);
    int newsize = size + sizeof(void *)*2 + sizeof(int);
    char *newmsg = (char*)CmiAlloc(newsize);
    memcpy(newmsg, m, size);
    memcpy(newmsg+size, &oldep, sizeof(int));
    memcpy(newmsg+size+sizeof(int), &slot->destAddress[0], sizeof(void *));
    memcpy(newmsg+size+sizeof(int)+sizeof(void*), &slot->destSizeAddress[0], sizeof(void *));
    CmiFree(m);
    CMI_MSG_SIZE(data) = size;
    CmiSetHandler(newmsg, persistentSendMsgHandlerIdx);
    phs = NULL; phsSize = 0;
    CmiSyncSendAndFree(slot->destPE, newsize, newmsg);
  }
  else {
#if 1
    /* buffer until ready */
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
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);

  /*  CmiPrintf("Setting root to %d\n", 0); */
  if (CmiMyPe()==destPE) {
#if CMI_QD
    CQdCreate(CpvAccess(cQdState), 1);
#endif
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiSendPersistentMsg(h, destPE, size, dupmsg);
}

/* called in PumpMsgs */
int PumpPersistent()
{
  PersistentReceivesTable *slot = persistentReceivesTableHead;
  int status = 0;
  while (slot) {
    unsigned int size = *(slot->recvSizePtr[0]);
    if (size > 0)
    {
      char *msg = slot->messagePtr[0];
/*CmiPrintf("[%d] size: %d rank:%d msg:%p %p\n", CmiMyPe(), size, CMI_DEST_RANK(msg), msg, slot->messagePtr);*/

#if 0
      void *dupmsg;
      dupmsg = CmiAlloc(size);
      
      _MEMCHECK(dupmsg);
      memcpy(dupmsg, msg, size);
      msg = dupmsg;
#else
      /* return messagePtr directly and user MUST make sure not to delete it. */
      /*CmiPrintf("[%d] %p size:%d rank:%d root:%d\n", CmiMyPe(), msg, size, CMI_DEST_RANK(msg), CMI_BROADCAST_ROOT(msg));*/

      CmiReference(msg);
#endif
      CmiPushPE(CMI_DEST_RANK_NET(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT(msg))
          SendSpanningChildrenNet(size, msg);
#endif
      *(slot->recvSizePtr[0]) = 0;
      status = 1;
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
    CmiFree(msg);
}

void persist_machine_init(void)
{
  persistentSendMsgHandlerIdx =
       CmiRegisterHandler((CmiHandler)sendPerMsgHandler);
}

void setupRecvSlot(PersistentReceivesTable *slot, int maxBytes)
{
  int i;
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
    char *buf = PerAlloc(maxBytes+sizeof(int)*2);
    _MEMCHECK(buf);
    memset(buf, 0, maxBytes+sizeof(int)*2);
    slot->messagePtr[i] = buf;
    slot->recvSizePtr[i] = (unsigned int*)CmiAlloc(sizeof(unsigned int));
    *(slot->recvSizePtr[0]) = 0;
  }
  slot->sizeMax = maxBytes;
}


