/*
  included in machine.c
  Nitin Bhat 06/03/2016
*/
#include "machine-rdma.h"
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
  void (*done_fn)(pami_context_t, void *, pami_result_t),
  int offset,
  pami_memregion_t *mregion,
  int size)
{
  pami_rget_simple_t  rget;
  rget.rma.dest    = origin;
  rget.rma.bytes   = size;
  rget.rma.cookie  = cookie;
  rget.rma.done_fn = done_fn;
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

//function to perform Rput
pami_result_t putData(
  pami_context_t       context,
  pami_endpoint_t      origin,
  void *buffer,
  void *cookie,
  int dst_context,
  void (*done_fn)(pami_context_t, void *, pami_result_t),
  int offset,
  pami_memregion_t *mregion,
  int size)
{
  pami_rput_simple_t  rput;
  rput.rma.dest    = origin;
  rput.rma.bytes   = size;
  rput.rma.cookie  = cookie;
  rput.rma.done_fn = done_fn;
  rput.rma.hints.buffer_registered = PAMI_HINT_ENABLE;
  rput.rma.hints.use_rdma = PAMI_HINT_ENABLE;
  rput.rdma.local.mr      = &cmi_pami_memregion[dst_context].mregion;
  rput.rdma.local.offset  = (size_t)buffer -
    (size_t)cmi_pami_memregion[dst_context].baseVA;
  rput.rdma.remote.mr     = mregion;
  rput.rdma.remote.offset = offset;

  MACHSTATE5(3, "[%d]putData, context:%p, size: %d, buffer: %p, cookie: %p\n", CmiMyPe(), context, size, buffer, cookie);

  int to_lock = 0;
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  to_lock = CpvAccess(uselock);
#endif

  if(to_lock)
    PAMIX_CONTEXT_LOCK(context);

  pami_result_t res = PAMI_Rput (context, &rput);

  if(to_lock)
    PAMIX_CONTEXT_UNLOCK(context);
  MACHSTATE2(3, "[%d]PAMI_Rput result: %d\n", CmiMyPe(), res);
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
          recvInfo->dstContext, rzv_rdma_recv_done, offset, &recvInfo->mregion, size);
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
  CmiRdmaAck *ack = *((void **) header_addr);
  ack->fnPtr(ack->token);
  //free callback structure, CmiRdmaAck allocated in CmiSetRdmaAck
  free(ack);
}

/* Support for Nocopy Direct API */

// Function called on completion of the direct rdma operation
void rzv_rdma_direct_recv_done (pami_context_t     ctxt,
    void             * clientdata,
    pami_result_t      result)
{
    DECR_ORECVS();
    // Call the ack handler function
    CmiInvokeNcpyAck(clientdata);
}

// Perform an RDMA Get call into the local target address from the remote source address
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  int size) {

  CmiAssert(srcAckSize == tgtAckSize);
  void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, tgtAddr, tgtAck, tgtPe, srcAckSize);

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  int c = CmiMyNode() % cmi_pami_numcontexts;
  pami_context_t my_context = cmi_pami_contexts[c];
#else
  pami_context_t my_context= MY_CONTEXT();
#endif

#if CMK_PAMI_MULTI_CONTEXT &&  CMK_NODE_QUEUE_AVAILABLE
  size_t dst_context = (CmiMyNode() != DGRAM_NODEMESSAGE) ? (CmiMyNode()>>LTPS) : (rand_r(&r_seed) % cmi_pami_numcontexts);
#else
  size_t dst_context = 0;
#endif

  CmiPAMIRzvRdmaPtr_t *src_Info = (CmiPAMIRzvRdmaPtr_t *)srcInfo;

  INCR_ORECVS();

  // Create end point for current node
  pami_endpoint_t origin;
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiNodeOf(srcPe), dst_context, &origin);

  getData(my_context, origin, (void *)tgtAddr, ref, dst_context, rzv_rdma_direct_recv_done, src_Info->offset, &src_Info->mregion, size);
}

// Perform an RDMA Put call into the remote target address from the local source address
void LrtsIssueRput(
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size) {

  CmiAssert(srcAckSize == tgtAckSize);
  void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, tgtAddr, tgtAck, tgtPe, srcAckSize);

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  int c = CmiMyNode() % cmi_pami_numcontexts;
  pami_context_t my_context = cmi_pami_contexts[c];
#else
  pami_context_t my_context= MY_CONTEXT();
#endif

#if CMK_PAMI_MULTI_CONTEXT &&  CMK_NODE_QUEUE_AVAILABLE
  size_t dst_context = (CmiMyNode() != DGRAM_NODEMESSAGE) ? (CmiMyNode()>>LTPS) : (rand_r(&r_seed) % cmi_pami_numcontexts);
#else
  size_t dst_context = 0;
#endif

  CmiPAMIRzvRdmaPtr_t *tgt_Info = (CmiPAMIRzvRdmaPtr_t *)tgtInfo;

  INCR_ORECVS();

  // Create end point for current node
  pami_endpoint_t origin;
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiNodeOf(tgtPe), dst_context, &origin);

  putData(my_context, origin, (void *)srcAddr, ref, dst_context, rzv_rdma_direct_recv_done, tgt_Info->offset, &tgt_Info->mregion, size);
}

// Method invoked to deregister target memory resources
void LrtsReleaseTargetResource(void *info, int pe){
  // Nothing to do as there is no explicit memory registration/deregistration
}

// Method invoked to deregister source memory resources
void LrtsReleaseSourceResource(void *info, int pe){
  // Nothing to do as there is no explicit memory registration/deregistration
}
