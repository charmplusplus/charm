/*
  included in machine.c
    Yanhua 01/27/2013
*/

/*
  machine specific persistent comm functions:
  * LrtsSendPersistentMsg
  * CmiSyncSendPersistent
  * PumpPersistent
  * PerAlloc PerFree      // persistent message memory allocation/free functions
  * persist_machine_init  // machine specific initialization call
*/
typedef struct _cmi_pami_rzv_persist {
  void             * srcPtr;
  void             * destPtr; 
  size_t           offset;
  int              bytes;
  int              dst_context;
}CmiPAMIRzvPersist_t;  



#define CMI_PAMI_RZV_PERSIST_DISPATCH            11 
void rzv_persist_pkt_dispatch (pami_context_t context, void *clientdata, const void *header_addr, size_t header_size, const void * pipe_addr,  size_t pipe_size,  pami_endpoint_t origin, pami_recv_t  * recv);

void _initPersistent( pami_context_t *contexts, int nc)
{
    pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
    pami_dispatch_callback_function pfn;
    int i = 0;
    for (i = 0; i < nc; ++i) {

        pfn.p2p = rzv_persist_pkt_dispatch;
        PAMI_Dispatch_set (cmi_pami_contexts[i],
        CMI_PAMI_RZV_PERSIST_DISPATCH,
        pfn,
        NULL,
        options);   
    }
}

void rzv_persist_recv_done   (pami_context_t     ctxt, 
    void             * clientdata, 
    pami_result_t      result) 
{
  CmiPAMIRzvRecv_t recv = *(CmiPAMIRzvRecv_t *)clientdata;
  CmiReference(recv.msg);
  recv_done(ctxt, recv.msg, PAMI_SUCCESS);
  sendAck(ctxt, &recv);
}

void rzv_persist_pkt_dispatch (pami_context_t       context,   
    void               * clientdata,
    const void         * header_addr,
    size_t               header_size,
    const void         * pipe_addr,  
    size_t               pipe_size,  
    pami_endpoint_t      origin,
    pami_recv_t         * recv) 
{
  INCR_ORECVS();    

  CmiPAMIRzvPersist_t  *rzv_hdr = (CmiPAMIRzvPersist_t *) header_addr;
  CmiAssert (header_size == sizeof(CmiPAMIRzvPersist_t));  
  int alloc_size = rzv_hdr->bytes;
  char * buffer  = rzv_hdr->destPtr; 
  //char * buffer  = (char *)CmiAlloc(alloc_size + sizeof(CmiPAMIRzvRecv_t));
  //char *buffer=(char*)CmiAlloc(alloc_size+sizeof(CmiPAMIRzvRecv_t)+sizeof(int))
  //*(int *)(buffer+alloc_size) = *(int *)header_addr;  
  CmiAssert (recv == NULL);

  CmiPAMIRzvRecv_t *rzv_recv = (CmiPAMIRzvRecv_t *)(buffer+alloc_size);
  rzv_recv->msg        = buffer;
  rzv_recv->src_ep     = origin;
  rzv_recv->src_buffer = rzv_hdr->srcPtr;

  CmiAssert (pipe_addr != NULL);
  pami_memregion_t *mregion = (pami_memregion_t *) pipe_addr;
  CmiAssert (pipe_size == sizeof(pami_memregion_t));

  //Rzv inj fifos are on the 17th core shared by all contexts
  pami_rget_simple_t  rget;
  rget.rma.dest    = origin;
  rget.rma.bytes   = rzv_hdr->bytes;
  rget.rma.cookie  = rzv_recv;
  rget.rma.done_fn = rzv_persist_recv_done;
  rget.rma.hints.buffer_registered = PAMI_HINT_ENABLE;
  rget.rma.hints.use_rdma = PAMI_HINT_ENABLE;
  rget.rdma.local.mr      = &cmi_pami_memregion[rzv_hdr->dst_context].mregion;  
  rget.rdma.local.offset  = (size_t)buffer - 
    (size_t)cmi_pami_memregion[rzv_hdr->dst_context].baseVA;
  rget.rdma.remote.mr     = mregion; //from message payload
  rget.rdma.remote.offset = rzv_hdr->offset;

  PAMI_Rget (context, &rget);  
}


void LrtsSendPersistentMsg(PersistentHandle h, int destNode, int size, void *msg)
{
    int         to_lock; 
    int         destIndex; 
    PersistentSendsTable *slot = (PersistentSendsTable *)h;
    if (h==NULL) {
        CmiAbort("LrtsSendPersistentMsg: not a valid PersistentHandle");
    }
    CmiAssert(CmiNodeOf(slot->destPE) == destNode);
    if (size > slot->sizeMax) {
        CmiPrintf("size: %d sizeMax: %d mype=%d destPe=%d\n", size, slot->sizeMax, CmiMyPe(), destNode);
        CmiAbort("Abort: Invalid size\n");
    }

    destIndex = slot->addrIndex;
    if (slot->destBuf[destIndex].destAddress) {
         //CmiPrintf("[%d===%d] LrtsSendPersistentMsg h=%p hdl=%d destNode=%d destAddress=%p size=%d\n", CmiMyPe(), destNode, h, CmiGetHandler(m), destNode, slot->destBuf[0].destAddress, size);

        slot->addrIndex = (destIndex+1)%PERSIST_BUFFERS_NUM;
#if  DELTA_COMPRESS
        if(slot->compressFlag)
        {
            size = CompressPersistentMsg(h, size, msg);
        }
#endif
  
        CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
        CMI_MSG_SIZE(msg) = size;
        CMI_SET_CHECKSUM(msg, size);
        to_lock = CpvAccess(uselock);
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
        int c = node % cmi_pami_numcontexts;
        pami_context_t context = cmi_pami_contexts[c];    
#else
        pami_context_t context = MY_CONTEXT();    
#endif

        pami_endpoint_t target;
#if CMK_PAMI_MULTI_CONTEXT
        size_t dst_context = myrand(&r_seed) % cmi_pami_numcontexts;
#else
        size_t dst_context = 0;
#endif
        PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)destNode, dst_context, &target);

        CmiPAMIRzvPersist_t   rzv;
        rzv.bytes       = size;
        rzv.srcPtr      = msg;
        rzv.destPtr     = slot->destBuf[destIndex].destAddress;
        rzv.offset      = (size_t)msg - (size_t)cmi_pami_memregion[0].baseVA;
        rzv.dst_context = dst_context;

        pami_send_immediate_t parameters;
        parameters.dispatch        = CMI_PAMI_RZV_PERSIST_DISPATCH;
        parameters.header.iov_base = &rzv;
        parameters.header.iov_len  = sizeof(rzv);
        parameters.data.iov_base   = &cmi_pami_memregion[0].mregion;      
        parameters.data.iov_len    = sizeof(pami_memregion_t);
        parameters.dest = target;

        if(to_lock)
            PAMIX_CONTEXT_LOCK(context);

        PAMI_Send_immediate (context, &parameters);

        if(to_lock)
            PAMIX_CONTEXT_UNLOCK(context);
    } else {

#if 1
        if (slot->messageBuf != NULL) {
            CmiPrintf("Unexpected message in buffer on %d\n", CmiMyPe());
            CmiAbort("");
        }
        slot->messageBuf = msg;
        slot->messageSize = size;
#else
    /* normal send */
        PersistentHandle  *phs_tmp = phs;
        int phsSize_tmp = phsSize;
        phs = NULL; phsSize = 0;
        CmiPrintf("[%d]Slot sending message directly\n", CmiMyPe());
        CmiSyncSendAndFree(slot->destPE, size, msg);
        phs = phs_tmp; phsSize = phsSize_tmp;
#endif
    }
}


extern void CmiReference(void *blk);

                                                                                
void PerFree(char *msg)
{
    CmiFree(msg);
}

/* machine dependent init call */
void persist_machine_init(void)
{
}

void initSendSlot(PersistentSendsTable *slot)
{
  int i;
  slot->destPE = -1;
  slot->sizeMax = 0;
  slot->destHandle = 0; 
  memset(&slot->destBuf, 0, sizeof(PersistentBuf)*PERSIST_BUFFERS_NUM);
  slot->messageBuf = 0;
  slot->messageSize = 0;
  slot->prev = slot->next = NULL;
}

void initRecvSlot(PersistentReceivesTable *slot)
{
  int i;
  memset(&slot->destBuf, 0, sizeof(PersistentBuf)*PERSIST_BUFFERS_NUM);
  slot->sizeMax = 0;
  //slot->index = -1;
  slot->prev = slot->next = NULL;
}

void setupRecvSlot(PersistentReceivesTable *slot, int maxBytes)
{
  int i;
  for (i=0; i<PERSIST_BUFFERS_NUM; i++) {
    char *buf = CmiAlloc(maxBytes+sizeof(CmiPAMIRzvRecv_t));
    _MEMCHECK(buf);
    memset(buf, 0, maxBytes+sizeof(CmiPAMIRzvRecv_t));
    slot->destBuf[i].destAddress = buf;
    /* note: assume first integer in converse header is the msg size */
    //slot->destBuf[i].destSizeAddress = (unsigned int*)buf;
    memset(buf, 0, maxBytes+sizeof(CmiPAMIRzvRecv_t));
  }
  slot->sizeMax = maxBytes;
  slot->addrIndex = 0;
}

void clearRecvSlot(PersistentReceivesTable *slot)
{
}

PersistentHandle getPersistentHandle(PersistentHandle h, int toindex)
{
    return h;
}
