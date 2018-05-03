/*
  included in machine.C
  Nitin Bhat 06/03/2016
*/
#define CMI_PAMI_ONESIDED_ACK_DISPATCH            12

void _initOnesided( pami_context_t *contexts, int nc){
  pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
  pami_dispatch_callback_function pfn;
  int i = 0;
  for (i = 0; i < nc; ++i) {
    pfn.p2p = ack_rdma_pkt_dispatch;
    PAMI_Dispatch_set (contexts[i],
      CMI_PAMI_ONESIDED_ACK_DISPATCH,
      pfn,
      NULL,
      options);
 }
}

//function called on completion of the rdma operation
void rzv_rdma_recv_done   (pami_context_t     ctxt,
    void             * clientdata,
    pami_result_t      result)
{
  CmiPAMIRzvRdmaRecvOp_t* recvOpInfo = (CmiPAMIRzvRdmaRecvOp_t *)clientdata;
  CmiPAMIRzvRdmaRecv_t* recvInfo = (CmiPAMIRzvRdmaRecv_t *)(
                                        ((char *)recvOpInfo)
                                      - recvOpInfo->opIndex * sizeof(CmiPAMIRzvRdmaRecvOp_t)
                                      - sizeof(CmiPAMIRzvRdmaRecv_t));

  rdma_sendAck(ctxt, recvOpInfo, recvInfo->src_ep);
  recvInfo->comOps++;
  if(recvInfo->comOps == recvInfo->numOps){
    recv_done(ctxt, recvInfo->msg, PAMI_SUCCESS);
  }
}

//function to perform Rget
pami_result_t getData(
  pami_context_t       context,
  pami_endpoint_t      origin,
  void *buffer,
  void *cookie,
  int dst_context,
  int offset,
  pami_memregion_t *mregion,
  int size)
{
  pami_rget_simple_t  rget;
  rget.rma.dest    = origin;
  rget.rma.bytes   = size;
  rget.rma.cookie  = cookie;
  rget.rma.done_fn = rzv_rdma_recv_done;
  rget.rma.hints.buffer_registered = PAMI_HINT_ENABLE;
  rget.rma.hints.use_rdma = PAMI_HINT_ENABLE;
  rget.rdma.local.mr      = &cmi_pami_memregion[dst_context].mregion;
  rget.rdma.local.offset  = (size_t)buffer -
    (size_t)cmi_pami_memregion[dst_context].baseVA;
  rget.rdma.remote.mr     = mregion;
  rget.rdma.remote.offset = offset;

  MACHSTATE5(3, "[%d]getData, context:%p, size: %d, buffer: %p, cookie: %p\n", CmiMyPe(), context, size, buffer, cookie);

  int to_lock = 0;
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  to_lock = CpvAccess(uselock);
#endif

  if(to_lock)
    PAMIX_CONTEXT_LOCK(context);

  pami_result_t res = PAMI_Rget (context, &rget);

  if(to_lock)
    PAMIX_CONTEXT_UNLOCK(context);
  MACHSTATE2(3, "[%d]PAMI_Rget result: %d\n", CmiMyPe(), res);
  return res;
}



void LrtsIssueRgets(void *recv, int pe){

  CmiPAMIRzvRdmaRecv_t* recvInfo = (CmiPAMIRzvRdmaRecv_t*)recv;

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  int c = CmiMyNode() % cmi_pami_numcontexts;
  pami_context_t my_context = cmi_pami_contexts[c];
#else
  pami_context_t my_context= MY_CONTEXT();
#endif

  INCR_ORECVS();


  pami_endpoint_t      origin;
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiNodeOf(pe), recvInfo->dstContext, &origin);
  recvInfo->src_ep = origin;

  int i;
  for(i=0; i<recvInfo->numOps; i++){
    void *buffer = recvInfo->rdmaOp[i].buffer;
    int offset = recvInfo->rdmaOp[i].offset;
    int size = recvInfo->rdmaOp[i].size;
    MACHSTATE5(3, "[%d]LrtsIssueRgets, recv:%p, origin: %u, dst_context: %d, offset: %d\n", CmiMyPe(), recvInfo, origin, recvInfo->dstContext, offset);
    getData(my_context, origin, buffer, &recvInfo->rdmaOp[i],
          recvInfo->dstContext, offset, &recvInfo->mregion, size);
  }
}

//function to send the acknowledgement to the sender
void  rdma_sendAck (
    pami_context_t      context,
    CmiPAMIRzvRdmaRecvOp_t* recvOpInfo,
    int src_ep)
{
  pami_send_immediate_t parameters;
  parameters.dispatch        = CMI_PAMI_ONESIDED_ACK_DISPATCH;
  parameters.header.iov_base = &recvOpInfo->src_info;
  parameters.header.iov_len  = sizeof(void *);
  parameters.data.iov_base   = NULL;
  parameters.data.iov_len    = 0;
  parameters.dest            = src_ep;
  PAMI_Send_immediate (context, &parameters);
}


// function called on the sender on receiving acknowledgement from the receiver to signal the completion of the rdma operation
void ack_rdma_pkt_dispatch (
    pami_context_t       context,
    void               * clientdata,
    const void         * header_addr,
    size_t               header_size,
    const void         * pipe_addr,
    size_t               pipe_size,
    pami_endpoint_t      origin,
    pami_recv_t         * recv)
{
  CmiAssert(sizeof(void *) == header_size);
  CmiRdmaAck *ack = *((CmiRdmaAck **) header_addr);
  ack->fnPtr(ack->token);
  //free callback structure, CmiRdmaAck allocated in CmiSetRdmaAck
  free(ack);
}
