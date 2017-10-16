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
    memregion = rdmaRecvOpInfo->mr;

    if(recvInfo->comOps == recvInfo->numOps) {
      // All the RDMA operations for one entry method have completed
      ofi_onesided_all_ops_done(recvInfo->msg);
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

void ofi_post_nocopy_read(CmiOfiRdmaRecvOp_t *rdmaRecvOpInfo) {
  int remaining     = rdmaRecvOpInfo->len;
  char *lbuf        = (char *)rdmaRecvOpInfo->buf;
  char *rbuf        = (FI_MR_SCALABLE == context.mr_mode) ? 0 : (void *)rdmaRecvOpInfo->src_buf;
  int nodeNo        = rdmaRecvOpInfo->src_nodeNo;
  uint64_t rkey     = rdmaRecvOpInfo->src_key;
  struct fid_mr *mr = rdmaRecvOpInfo->mr;
  size_t chunk_size;
  OFIRequest *rma_req;

  while (remaining > 0) {
    chunk_size = (remaining <= context.rma_maxsize) ? remaining : context.rma_maxsize;

#if USE_OFIREQUEST_CACHE
    rma_req = alloc_request(context.request_cache);
#else
    rma_req = CmiAlloc(sizeof(OFIRequest));
#endif

    CmiAssert(rma_req);
    rma_req->callback = ofi_onesided_read_callback;
    rma_req->data.rma_ncpy_info = (void *)rdmaRecvOpInfo;

    rdmaRecvOpInfo->completion_count++;

    OFI_RETRY(fi_read(context.ep,
                      lbuf,
                      chunk_size,
                      (mr) ? fi_mr_desc(mr) : NULL,
                      nodeNo,
                      (uint64_t)rbuf,
                      rkey,
                      &rma_req->context));

    remaining -= chunk_size;
    lbuf      += chunk_size;
    rbuf      += chunk_size;
  }
}

void LrtsIssueRgets(void *recv, int pe) {
  CmiOfiRdmaRecv_t* recvInfo = (CmiOfiRdmaRecv_t *)recv;
  int i;
  for(i = 0; i < recvInfo->numOps; i++) {
    ofi_post_nocopy_read(&(recvInfo->rdmaOp[i]));
  }
}
