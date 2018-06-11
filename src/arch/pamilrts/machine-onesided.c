/*
  included in machine.C
  Nitin Bhat 06/03/2016
*/

#define CMI_PAMI_DIRECT_GET_DISPATCH              11
#define CMI_PAMI_ONESIDED_ACK_DISPATCH            12

void _initOnesided( pami_context_t *contexts, int nc){
  pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
  pami_dispatch_callback_function pfn;
  int i = 0;
  for (i = 0; i < nc; ++i) {
    pfn.p2p = rdma_direct_get_dispatch;
    PAMI_Dispatch_set (contexts[i],
      CMI_PAMI_DIRECT_GET_DISPATCH,
      pfn,
      NULL,
      options);
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
#if CMK_BLUEGENEQ
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

#else
  size_t bytes_out;
  pami_memregion_t local_mregion;

  PAMI_Memregion_create(context,
                        buffer,
                        size,
                        &bytes_out,
                        &local_mregion); 

  pami_get_simple_t get;
  memset(&get, 0, sizeof(get));
  get.rma.dest = origin;
  get.rma.bytes = size;
  get.rma.cookie = cookie;
  get.rma.done_fn = done_fn;
  get.rma.hints.buffer_registered = PAMI_HINT_ENABLE;
  get.rma.hints.use_rdma = PAMI_HINT_ENABLE;
  get.rma.hints.use_shmem = PAMI_HINT_DEFAULT;
  get.rma.hints.remote_async_progress = PAMI_HINT_DEFAULT;
  get.addr.local = buffer;
  get.addr.remote = (void *)offset;

  int to_lock = 0;
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  to_lock = CpvAccess(uselock);
#endif

  if(to_lock)
    PAMIX_CONTEXT_LOCK(context);

  pami_result_t res = PAMI_Get (context, &get);

  if(to_lock)
    PAMIX_CONTEXT_UNLOCK(context);
  MACHSTATE2(3, "[%d]PAMI_Get result: %d\n", CmiMyPe(), res);
  return res;
#endif
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

/* Support for Nocopy Direct API */

// Function called on completion of the receive operation for ncpyOpInfo object
// This method uses the ncpyOpInfo object and performs a Get operation instead of PUT
void ncpyOpInfo_recv_done(pami_context_t ctxt, void *clientdata, pami_result_t result){

  DECR_ORECVS();

  // perform GET instead of PUT
  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)clientdata;

  resetNcpyOpInfoPointers(ncpyOpInfo);

#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  CpvAccess(uselock) = 0;
#endif
  LrtsIssueRget(ncpyOpInfo);
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  CpvAccess(uselock) = 1;
#endif
}

// Function called to receive a ncpyOpInfo object in order to perform a GET operation instead of PUT
// Currently, on BGQ, I am seeing a weird error while performing PUT operations
void rdma_direct_get_dispatch (
    pami_context_t       context,
    void               * clientdata,
    const void         * header_addr,
    size_t               header_size,
    const void         * pipe_addr,
    size_t               pipe_size,
    pami_endpoint_t      origin,
    pami_recv_t         * recv)
{
  INCR_ORECVS();
  int alloc_size = pipe_size;
  char *buffer = (char *)CmiAlloc(alloc_size);

  if(recv) {
    recv->local_fn = ncpyOpInfo_recv_done;
    recv->cookie   = buffer;
    recv->type     = PAMI_TYPE_BYTE;
    recv->addr     = buffer;
    recv->offset   = 0;
    recv->data_fn  = PAMI_DATA_COPY;
  } else {
    memcpy(buffer, pipe_addr, pipe_size);
    ncpyOpInfo_recv_done(NULL, buffer, PAMI_SUCCESS);
  }
}

// Function called on completion of the direct rdma operation
void rzv_rdma_direct_recv_done (pami_context_t     ctxt,
    void             * clientdata,
    pami_result_t      result)
{
  DECR_ORECVS();

#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  CpvAccess(uselock) = 0;
#endif
  // Call the ack handler function
  CmiInvokeNcpyAck(clientdata);
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  CpvAccess(uselock) = 1;
#endif
}

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo) {

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

  CmiPAMIRzvRdmaPtr_t *src_Info = (CmiPAMIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize());

  INCR_ORECVS();

  // Create end point for current node
  pami_endpoint_t origin;
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiNodeOf(ncpyOpInfo->srcPe), dst_context, &origin);

  getData(my_context, origin, (void *)(ncpyOpInfo->destPtr), ncpyOpInfo, dst_context, rzv_rdma_direct_recv_done, src_Info->offset, &src_Info->mregion, ncpyOpInfo->srcSize);
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo) {

  // Create end point for destination node
  pami_endpoint_t target;
  int node = CmiNodeOf(ncpyOpInfo->destPe);

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  int c = CmiMyNode() % cmi_pami_numcontexts;
  pami_context_t my_context = cmi_pami_contexts[c];
#else
  pami_context_t my_context= MY_CONTEXT();
#endif

#if CMK_PAMI_MULTI_CONTEXT &&  CMK_NODE_QUEUE_AVAILABLE
  size_t dst_context = (node != DGRAM_NODEMESSAGE) ? (node>>LTPS) : (rand_r(&r_seed) % cmi_pami_numcontexts);
#else
  size_t dst_context = 0;
#endif

  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)node, dst_context, &target);

  // Send a message using the Eager protocol to send the ncpyOpInfo object
  // This allows the receiver process to perform a Get instead of Put
  // Get is used instead of the native Put because, on BGQ, I am seeing a weird error
  // while performing PUT operations
  pami_send_t parameters;
  parameters.send.dispatch        = CMI_PAMI_DIRECT_GET_DISPATCH;
  parameters.send.header.iov_base = NULL;
  parameters.send.header.iov_len  = 0;
  parameters.send.data.iov_base   = ncpyOpInfo;
  parameters.send.data.iov_len    = ncpyOpInfo->ncpyOpInfoSize;
  parameters.events.cookie        = ncpyOpInfo;
  parameters.events.local_fn      = send_done;
  parameters.events.remote_fn     = NULL;
  memset(&parameters.send.hints, 0, sizeof(parameters.send.hints));
  parameters.send.dest = target;

  // Send my info to the other node for it to perform a GET operation
  int to_lock = 0;
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
  to_lock = CpvAccess(uselock);
#endif

  if(to_lock)
    PAMIX_CONTEXT_LOCK(my_context);

  INCR_MSGQLEN();
  PAMI_Send (my_context, &parameters);

  if(to_lock)
    PAMIX_CONTEXT_UNLOCK(my_context);
}

// Method invoked to deregister memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode) {
  // Nothing to do as there is no explicit memory registration/deregistration
}
