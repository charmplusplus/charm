void ofi_onesided_all_ops_done(char *msg) {
  int stdlen = ((CmiMsgHeaderBasic *) msg)->size;
  // Invoke the message handler
  handleOneRecvedMsg(stdlen, msg);
}

void process_onesided_completion_ack(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
  struct fid *mr;
  OfiRdmaAck_t *ofiAck = (OfiRdmaAck_t *)req->data.rma_ncpy_ack;
  CmiRdmaAck *ack = (CmiRdmaAck *)(ofiAck->src_ref);
  // Invoke the ack handler function
  ack->fnPtr(ack->token);

  // Deregister the buffer
  mr = (struct fid *)(ofiAck->src_mr);
  CmiAssert(mr);
  fi_close(mr);
}

struct fid_mr* registerDirectMemory(const void *addr, int size) {
  struct fid_mr *mr;
  uint64_t requested_key = 0;
  int ret;

  if(FI_MR_SCALABLE == context.mr_mode) {
    requested_key = __sync_fetch_and_add(&(context.mr_counter), 1);
  }
  ret = fi_mr_reg(context.domain,
                  addr,
                  size,
                  FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE,
                  0ULL,
                  requested_key,
                  0ULL,
                  &mr,
                  NULL);
  if (ret) {
    CmiAbort("registerDirectMemory: fi_mr_reg failed!\n");
  }
  return mr;
}


static inline void ofi_onesided_send_ack_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
#if USE_OFIREQUEST_CACHE
  free_request(req);
#else
  CmiFree(req);
#endif
}

static inline void ofi_onesided_read_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
  CmiOfiRdmaRecvOp_t *rdmaRecvOpInfo = (CmiOfiRdmaRecvOp_t *)(req->data.rma_ncpy_info);
  rdmaRecvOpInfo->completion_count--;
  struct fid* memregion;

  if(0 == rdmaRecvOpInfo->completion_count) {
    CmiOfiRdmaRecv_t *recvInfo = (CmiOfiRdmaRecv_t *)((char *)rdmaRecvOpInfo
                        - rdmaRecvOpInfo->opIndex * sizeof(CmiOfiRdmaRecvOp_t)
                        - sizeof(CmiOfiRdmaRecv_t));

    req->callback      = ofi_onesided_send_ack_callback;

    // Send an acknowledgement message to the sender to indicate the completion of the RDMA operation
    ofi_send(&(rdmaRecvOpInfo->ack),
             sizeof(OfiRdmaAck_t),
             rdmaRecvOpInfo->src_nodeNo,
             OFI_RDMA_OP_ACK,
             req);

    recvInfo->comOps++;

    // Store the memregion for registration
    memregion = (struct fid *)rdmaRecvOpInfo->mr;

    if(recvInfo->comOps == recvInfo->numOps) {
      // All the RDMA operations for one entry method have completed
      ofi_onesided_all_ops_done((char*)recvInfo->msg);
    }

    // Deregister the memory region
    if(memregion)
      fi_close(memregion);

  } else {
#if USE_OFIREQUEST_CACHE
    free_request(req);
#else
    CmiFree(req);
#endif
  }
}

void ofi_post_nocopy_operation(
  char *lbuf,
  const char *rbuf,
  int  remoteNodeNo,
  uint64_t rkey,
  int size,
  struct fid_mr *lmr,
  ofiCallbackFn done_fn,
  void *cbRef,
  int  *completion_count,
  int operation) {

  int remaining = size;
  size_t chunk_size;
  OFIRequest *rma_req;

  while (remaining > 0) {
    chunk_size = (remaining <= context.rma_maxsize) ? remaining : context.rma_maxsize;

#if USE_OFIREQUEST_CACHE
    rma_req = alloc_request(context.request_cache);
#else
    rma_req = (OFIRequest*)CmiAlloc(sizeof(OFIRequest));
#endif

    CmiAssert(rma_req);
    rma_req->callback = done_fn;
    rma_req->data.rma_ncpy_info = cbRef;

    (*completion_count)++;

    if(operation == OFI_READ_OP) {
      // Perform an RDMA read or get operation
      OFI_RETRY(fi_read(context.ep,
                        lbuf,
                        chunk_size,
                        (lmr) ? fi_mr_desc(lmr) : NULL,
                        remoteNodeNo,
                        (uint64_t)rbuf,
                        rkey,
                        &rma_req->context));
    } else if(operation == OFI_WRITE_OP) {
      // Perform an RDMA write or put operation
      OFI_RETRY(fi_write(context.ep,
                        lbuf,
                        chunk_size,
                        (lmr) ? fi_mr_desc(lmr) : NULL,
                        remoteNodeNo,
                        (uint64_t)rbuf,
                        rkey,
                        &rma_req->context));
    } else {
      CmiAbort("ofi_post_nocopy_operation: Invalid RDMA operation\n");
    }

    remaining -= chunk_size;
    lbuf      += chunk_size;
    rbuf      += chunk_size;
  }
}

void LrtsIssueRgets(void *recv, int pe) {
  CmiOfiRdmaRecv_t* recvInfo = (CmiOfiRdmaRecv_t *)recv;
  int i;
  for(i = 0; i < recvInfo->numOps; i++) {
    CmiOfiRdmaRecvOp_t *rdmaRecvOpInfo = &(recvInfo->rdmaOp[i]);
    const char *rbuf        = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (const char*)rdmaRecvOpInfo->src_buf;
    ofi_post_nocopy_operation(
        (char *)rdmaRecvOpInfo->buf,
        rbuf,
        rdmaRecvOpInfo->src_nodeNo,
        rdmaRecvOpInfo->src_key,
        rdmaRecvOpInfo->len,
        rdmaRecvOpInfo->mr,
        ofi_onesided_read_callback,
        (void *)rdmaRecvOpInfo,
        &(rdmaRecvOpInfo->completion_count),
        OFI_READ_OP);
  }
}

/* Support for Nocopy Direct API */
// Method called on the completion of a direct onesided operation

// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){
  CmiOfiRdmaPtr_t *rdmaDest = (CmiOfiRdmaPtr_t *)info;
  uint64_t requested_key = 0;
  int ret;

  /* Register the destination buffer */
  if(FI_MR_SCALABLE == context.mr_mode) {
    requested_key = __sync_fetch_and_add(&(context.mr_counter), 1);
  }
  ret = fi_mr_reg(context.domain,
                  ptr,
                  size,
                  FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE,
                  0ULL,
                  requested_key,
                  0ULL,
                  &(rdmaDest->mr),
                  NULL);
  if (ret) {
    CmiAbort("LrtsSetRdmaBufferInfo: fi_mr_reg failed!\n");
  }
  rdmaDest->key = fi_mr_key(rdmaDest->mr);
}

static inline void ofi_onesided_direct_operation_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
  CmiOfiRdmaComp_t *rdmaComp = (CmiOfiRdmaComp_t *)(req->data.rma_ncpy_info);
  rdmaComp->completion_count--;
  if(0 == rdmaComp->completion_count) {

    // Invoke the ack handler
    CmiInvokeNcpyAck(rdmaComp->ack_info);
    free(rdmaComp);
  }
#if USE_OFIREQUEST_CACHE
  free_request(req);
#else
  CmiFree(req);
#endif
}

void process_onesided_reg_and_put(struct fi_cq_tagged_entry *e, OFIRequest *req) {

  char *data = (char *)req->data.rma_ncpy_ack;

  // Allocate a new receiver buffer to receive other messages
  req->data.recv_buffer = CmiAlloc(context.eager_maxsize);

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(data);
  resetNcpyOpInfoPointers(ncpyOpInfo);

  // Do not free as this message
  ncpyOpInfo->freeMe = 0;

  struct fid_mr *mr = registerDirectMemory(ncpyOpInfo->srcPtr, ncpyOpInfo->size);
  const char *rbuf  = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (const char*)(ncpyOpInfo->destPtr);

  // Allocate a completion object for tracking write completion and ack handling
  CmiOfiRdmaComp_t* rdmaComp = (CmiOfiRdmaComp_t *)malloc(sizeof(CmiOfiRdmaComp_t));
  rdmaComp->ack_info         = ncpyOpInfo;
  rdmaComp->completion_count = 0;

  ofi_post_nocopy_operation(
      (char*)(ncpyOpInfo->srcPtr),
      rbuf,
      CmiNodeOf(ncpyOpInfo->destPe),
      ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->destLayerInfo))->key,
      ncpyOpInfo->size,
      mr,
      ofi_onesided_direct_operation_callback,
      (void *)rdmaComp,
      &(rdmaComp->completion_count),
      OFI_WRITE_OP);
}

void process_onesided_reg_and_get(struct fi_cq_tagged_entry *e, OFIRequest *req) {

  char *data = (char *)req->data.rma_ncpy_ack;

  // Allocate a new receiver buffer to receive other messages
  req->data.recv_buffer = CmiAlloc(context.eager_maxsize);

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(data);
  resetNcpyOpInfoPointers(ncpyOpInfo);

  struct fid_mr *mr = registerDirectMemory(ncpyOpInfo->destPtr, ncpyOpInfo->size);
  const char *rbuf  = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (const char*)(ncpyOpInfo->srcPtr);

  // Allocate a completion object for tracking write completion and ack handling
  CmiOfiRdmaComp_t* rdmaComp = (CmiOfiRdmaComp_t *)malloc(sizeof(CmiOfiRdmaComp_t));
  rdmaComp->ack_info         = ncpyOpInfo;
  rdmaComp->completion_count = 0;

  ofi_post_nocopy_operation(
      (char*)(ncpyOpInfo->destPtr),
      rbuf,
      CmiNodeOf(ncpyOpInfo->srcPe),
      ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->srcLayerInfo))->key,
      ncpyOpInfo->size,
      mr,
      ofi_onesided_direct_operation_callback,
      (void *)rdmaComp,
      &(rdmaComp->completion_count),
      OFI_READ_OP);
}


// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(
  NcpyOperationInfo *ncpyOpInfo,
  unsigned short int *srcMode,
  unsigned short int *destMode) {

  OFIRequest *req;

  // Register local buffer if it is not registered
  if(*destMode == CMK_BUFFER_UNREG) {
    CmiOfiRdmaPtr_t *dest_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->destLayerInfo);
    dest_info->mr = registerDirectMemory(ncpyOpInfo->destPtr, ncpyOpInfo->size);
    dest_info->key = fi_mr_key(dest_info->mr);
    *destMode = CMK_BUFFER_REG;

    // set registration info in the origDestLayerInfoPtr
    ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->origDestLayerInfoPtr))->mr = dest_info->mr;
    ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->origDestLayerInfoPtr))->key = dest_info->key;
  }

  if(*srcMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform PUT
#if USE_OFIREQUEST_CACHE
    req = alloc_request(context.request_cache);
#else
    req = (OFIRequest*)CmiAlloc(sizeof(OFIRequest));
#endif
    CmiAssert(req);

    ZERO_REQUEST(req);

    req->destNode = CmiNodeOf(ncpyOpInfo->srcPe);
    req->destPE   = ncpyOpInfo->srcPe;
    req->size     = ncpyOpInfo->ncpyOpInfoSize;
    req->callback = send_short_callback;
    req->data.short_msg = ncpyOpInfo;

    ofi_send(ncpyOpInfo,
             ncpyOpInfo->ncpyOpInfoSize,
             CmiNodeOf(ncpyOpInfo->srcPe),
             OFI_RDMA_DIRECT_REG_AND_PUT,
             req);
  } else {

    CmiOfiRdmaPtr_t *dest_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->destLayerInfo);
    CmiOfiRdmaPtr_t *src_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->srcLayerInfo);

    const char *rbuf        = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (const char*)(ncpyOpInfo->srcPtr);

    // Allocate a completion object for tracking read completion and ack handling
    CmiOfiRdmaComp_t* rdmaComp = (CmiOfiRdmaComp_t *)malloc(sizeof(CmiOfiRdmaComp_t));
    rdmaComp->ack_info         = ncpyOpInfo;
    rdmaComp->completion_count = 0;

    ofi_post_nocopy_operation(
        (char*)(ncpyOpInfo->destPtr),
        rbuf,
        CmiNodeOf(ncpyOpInfo->srcPe),
        src_info->key,
        ncpyOpInfo->size,
        dest_info->mr,
        ofi_onesided_direct_operation_callback,
        (void *)rdmaComp,
        &(rdmaComp->completion_count),
        OFI_READ_OP);
  }
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(
  NcpyOperationInfo *ncpyOpInfo,
  unsigned short int *srcMode,
  unsigned short int *destMode) {

  OFIRequest *req;

  // Register local buffer if it is not registered
  if(*srcMode == CMK_BUFFER_UNREG) {
    CmiOfiRdmaPtr_t *src_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->srcLayerInfo);
    src_info->mr = registerDirectMemory(ncpyOpInfo->srcPtr, ncpyOpInfo->size);
    src_info->key = fi_mr_key(src_info->mr);
    *srcMode = CMK_BUFFER_REG;

    // set registration info in the origSrcLayerInfoPtr
    ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->origSrcLayerInfoPtr))->mr = src_info->mr;
    ((CmiOfiRdmaPtr_t *)(ncpyOpInfo->origSrcLayerInfoPtr))->key = src_info->key;
  }

  if(*destMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform PUT
#if USE_OFIREQUEST_CACHE
    req = alloc_request(context.request_cache);
#else
    req = (OFIRequest*)CmiAlloc(sizeof(OFIRequest));
#endif
    CmiAssert(req);

    ZERO_REQUEST(req);

    req->destNode = CmiNodeOf(ncpyOpInfo->destPe);
    req->destPE   = ncpyOpInfo->destPe;
    req->size     = ncpyOpInfo->ncpyOpInfoSize;
    req->callback = send_short_callback;
    req->data.short_msg = ncpyOpInfo;

    ofi_send(ncpyOpInfo,
             ncpyOpInfo->ncpyOpInfoSize,
             CmiNodeOf(ncpyOpInfo->destPe),
             OFI_RDMA_DIRECT_REG_AND_GET,
             req);
  } else {

    CmiOfiRdmaPtr_t *dest_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->destLayerInfo);
    CmiOfiRdmaPtr_t *src_info = (CmiOfiRdmaPtr_t *)(ncpyOpInfo->srcLayerInfo);

    const char *rbuf        = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (const char*)(ncpyOpInfo->destPtr);

    // Allocate a completion object for tracking write completion and ack handling
    CmiOfiRdmaComp_t* rdmaComp = (CmiOfiRdmaComp_t *)malloc(sizeof(CmiOfiRdmaComp_t));
    rdmaComp->ack_info         = ncpyOpInfo;
    rdmaComp->completion_count = 0;

    ofi_post_nocopy_operation(
        (char*)(ncpyOpInfo->srcPtr),
        rbuf,
        CmiNodeOf(ncpyOpInfo->destPe),
        dest_info->key,
        ncpyOpInfo->size,
        src_info->mr,
        ofi_onesided_direct_operation_callback,
        (void *)rdmaComp,
        &(rdmaComp->completion_count),
        OFI_WRITE_OP);
  }
}

// Method invoked to deregister memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
  if(mode == CMK_BUFFER_REG) {
    CmiOfiRdmaPtr_t *rdmaSrc = (CmiOfiRdmaPtr_t *)info;
    int ret;

    // Deregister the buffer
    if(rdmaSrc->mr) {
      ret = fi_close((struct fid *)rdmaSrc->mr);
      if(ret)
        CmiAbort("LrtsDeregisterMem: fi_close(mr) failed!\n");
    }
  }
}
