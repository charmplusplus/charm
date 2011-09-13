/** @file
 * Elan persistent communication
 * @ingroup Machine
*/

/*
  included in machine.c
  Gengbin Zheng, 9/6/2011
*/

/*
  machine specific persistent comm functions:
  * LrtsSendPersistentMsg
  * CmiSyncSendPersistent
  * PumpPersistent
  * PerAlloc PerFree      // persistent message memory allocation/free functions
  * persist_machine_init  // machine specific initialization call
*/


void LrtsSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m)
{
    gni_post_descriptor_t *pd;
    gni_return_t status;
    RDMA_REQUEST        *rdma_request_msg;
    
    CmiAssert(h!=NULL);
    PersistentSendsTable *slot = (PersistentSendsTable *)h;
    CmiAssert(slot->used == 1);
    CmiAssert(slot->destPE == destPE);
    if (size > slot->sizeMax) {
        CmiPrintf("size: %d sizeMax: %d\n", size, slot->sizeMax);
        CmiAbort("Abort: Invalid size\n");
    }

/*CmiPrintf("[%d] LrtsSendPersistentMsg h=%p hdl=%d destPE=%d destAddress=%p size=%d\n", CmiMyPe(), *phs, CmiGetHandler(m), destPE, slot->destAddress[0], size);*/

    if (slot->destBuf[0].destAddress) {
        // uGNI part
     
        MallocPostDesc(pd);
        if(size < LRTS_GNI_RDMA_THRESHOLD) 
            pd->type            = GNI_POST_FMA_PUT;
        else
            pd->type            = GNI_POST_RDMA_PUT;

        pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
        pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd->length          = size;
        pd->local_addr      = (uint64_t) m;
        pd->local_mem_hndl  = GetMemHndl(m) ;
        pd->remote_addr     = (uint64_t)slot->destBuf[0].destAddress;
        pd->remote_mem_hndl = slot->destBuf[0].mem_hndl;
        pd->src_cq_hndl     = 0;//post_tx_cqh;     /* smsg_tx_cqh;  */
        pd->rdma_mode       = 0;

        if(pd->type == GNI_POST_RDMA_PUT) 
            status = GNI_PostRdma(ep_hndl_array[destPE], pd);
        else
            status = GNI_PostFma(ep_hndl_array[destPE],  pd);
        if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
        {
            MallocRdmaRequest(rdma_request_msg);
            rdma_request_msg->next = 0;
            rdma_request_msg->destNode = destPE;
            rdma_request_msg->pd = pd;
            if(pending_rdma_head == 0)
            {
                pending_rdma_head = rdma_request_msg;
            }else
            {
                pending_rdma_tail->next = rdma_request_msg;
            }
            pending_rdma_tail = rdma_request_msg;
        }else
            GNI_RC_CHECK("AFter posting", status);
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

#if 0
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
    LrtsSendPersistentMsg(h, destPE, size, dupmsg);
}
#endif

extern void CmiReference(void *blk);

#if 0

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

#endif

void *PerAlloc(int size)
{
  return CmiAlloc(size);
}
                                                                                
void PerFree(char *msg)
{
  //elan_CmiStaticFree(msg);
  CmiFree(msg);
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
    slot->destBuf[i].destAddress = buf;
    /* note: assume first integer in elan converse header is the msg size */
    slot->destBuf[i].destSizeAddress = (unsigned int*)buf;
    slot->destBuf[i].mem_hndl = GetMemHndl(buf);
  }
  slot->sizeMax = maxBytes;
}


