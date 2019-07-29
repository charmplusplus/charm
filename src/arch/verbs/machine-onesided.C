/*
 * included in machine-ibverbs.C
 * Jaemin Choi
 */

void postRdma(
  uint64_t local_addr,
  uint32_t local_rkey,
  uint64_t remote_addr,
  uint32_t remote_rkey,
  int size,
  int peNum,
  uint64_t rdmaPacket,
  int opcode) {

  CmiAssert(opcode == IBV_WR_RDMA_WRITE || opcode == IBV_WR_RDMA_READ);

  struct ibv_sge list;
  struct ibv_send_wr *bad_wr;
  struct ibv_send_wr wr;

  memset(&list, 0, sizeof(list));
  list.addr = (uintptr_t)local_addr;
  list.length = size;
  list.lkey = local_rkey;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = rdmaPacket;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = (ibv_wr_opcode)opcode;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;

  OtherNode node = nodes_by_pe[peNum];
#if CMK_IBVERBS_TOKENS_FLOW
  if(context->tokensLeft<0){
    char errMsg[200];
    sprintf(errMsg, "No remaining tokens! Pass a larger value for maxTokens (%d) using argument +IBVMaxSendTokens\n", maxTokens);
    CmiAbort(errMsg);
  }
#endif

  int retval;
  if (retval = ibv_post_send(node->infiData->qp, &wr, &bad_wr)) {
    char errMsg[200];
    CmiPrintf(" Pe:%d, Node:%d, thread id:%d infidata nodeno:[%d] failed with return value %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), node->infiData->nodeNo, retval);
    sprintf(errMsg,"ibv_post_send failed in postRdma!! Try passing a larger value for maxTokens (%d) using argument +IBVMaxSendTokens\n",maxTokens);
    CmiAbort(errMsg);
  }
}

/* Support for Nocopy Direct API */

// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){
  struct ibv_mr *mr;
  if(mode == CMK_BUFFER_PREREG) {
    mr = METADATAFIELD(ptr)->key;
  } else {
    mr = ibv_reg_mr(context->pd, (void *)ptr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  }
  if (!mr) {
    CmiAbort("Memory Registration Failed in LrtsSetRdmaBufferInfo!\n");
  }
  CmiVerbsRdmaPtr_t *rdmaInfo = (CmiVerbsRdmaPtr_t *)info;
  rdmaInfo->mr = mr;
  rdmaInfo->key = mr->rkey;
}

void registerDirectMemory(void *info, const void *addr, int size) {
  struct ibv_mr *mr = ibv_reg_mr(context->pd, (void *)addr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    CmiAbort("Memory Registration inside registerDirectMemory!\n");
  }
  CmiVerbsRdmaPtr_t *rdmaInfo = (CmiVerbsRdmaPtr_t *)info;
  rdmaInfo->mr = mr;
  rdmaInfo->key = mr->rkey;
}

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo) {

  if(ncpyOpInfo->isSrcRegistered == 0) {

    int acqLock = 0;

    // Lock around sending the small message
#if CMK_SMP
    LOCK_AND_SET();
#endif

    // Remote buffer is unregistered, send a message to register it and perform PUT
    infiPacket packet;
    MallocInfiPacket(packet);

    packet->size = ncpyOpInfo->ncpyOpInfoSize;
    packet->buf  = (char *)ncpyOpInfo;
    packet->header.code = INFIRDMA_DIRECT_REG_AND_PUT;
    packet->ogm  = NULL;

    struct ibv_mr *packetKey;
    if(ncpyOpInfo->opMode == CMK_DIRECT_API) {
      packetKey = METADATAFIELD(ncpyOpInfo)->key;
    } else if(ncpyOpInfo->opMode == CMK_EM_API || ncpyOpInfo->opMode == CMK_BCAST_EM_API) {
      // Register the small message in order to send it to the other side
      packetKey = ibv_reg_mr(context->pd, (void *)ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      if (!packetKey) {
        CmiAbort("Memory Registration Failed in LrtsIssueRget!\n");
      }
      // set opMode to for Reverse operation
      setReverseModeForNcpyOpInfo(ncpyOpInfo);
    }

    OtherNode node = &nodes[CmiNodeOf(ncpyOpInfo->srcPe)];
    EnqueuePacket(node, packet, ncpyOpInfo->ncpyOpInfoSize, packetKey);

#if CMK_SMP
    UNLOCK_AND_UNSET();
#endif

  } else {

    struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)malloc(sizeof(struct infiRdmaPacket));
    rdmaPacket->type = INFI_ONESIDED_DIRECT;
    rdmaPacket->localBuffer = ncpyOpInfo;

    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize());
    CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize());

    postRdma((uint64_t)(ncpyOpInfo->destPtr),
            dest_info->key,
            (uint64_t)(ncpyOpInfo->srcPtr),
            src_info->key,
            std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
            ncpyOpInfo->srcPe,
            (uint64_t)rdmaPacket,
            IBV_WR_RDMA_READ);
  }
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo) {

  if(ncpyOpInfo->isDestRegistered == 0) {

    int acqLock = 0;

    // Lock around sending the small message
#if CMK_SMP
    LOCK_AND_SET();
#endif

    // Remote buffer is unregistered, send a message to register it and perform GET
    infiPacket packet;
    MallocInfiPacket(packet);

    packet->size = ncpyOpInfo->ncpyOpInfoSize;
    packet->buf  = (char *)ncpyOpInfo;
    packet->header.code = INFIRDMA_DIRECT_REG_AND_GET;
    packet->ogm  = NULL;

    struct ibv_mr *packetKey = METADATAFIELD(ncpyOpInfo)->key;
    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize());
    OtherNode node = &nodes[CmiNodeOf(ncpyOpInfo->destPe)];
    EnqueuePacket(node, packet, ncpyOpInfo->ncpyOpInfoSize, packetKey);

#if CMK_SMP
    UNLOCK_AND_UNSET();
#endif

  } else {
    struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)malloc(sizeof(struct infiRdmaPacket));
    rdmaPacket->type = INFI_ONESIDED_DIRECT;
    rdmaPacket->localBuffer = ncpyOpInfo;

    CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize());
    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize());

    postRdma((uint64_t)(ncpyOpInfo->srcPtr),
            src_info->key,
            (uint64_t)(ncpyOpInfo->destPtr),
            dest_info->key,
            std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
            ncpyOpInfo->destPe,
            (uint64_t)rdmaPacket,
            IBV_WR_RDMA_WRITE);
  }
}

// Method invoked to deregister a memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
  CmiVerbsRdmaPtr_t *rdmadest = (CmiVerbsRdmaPtr_t *)info;

  if(mode != CMK_BUFFER_PREREG && mode != CMK_BUFFER_NOREG) {
    if (ibv_dereg_mr(rdmadest->mr)) {
      CmiAbort("ibv_dereg_mr() failed at LrtsDeregisterMem\n");
    }
  }
}

void LrtsInvokeRemoteDeregAckHandler(int pe, NcpyOperationInfo *ncpyOpInfo) {

  if(ncpyOpInfo->opMode == CMK_BCAST_EM_API)
    return;

  // Send a message to de-register remote buffer and invoke callback
  infiPacket packet;
  MallocInfiPacket(packet);

  struct ibv_mr *packetKey;
  NcpyOperationInfo *newNcpyOpInfo;

  if(ncpyOpInfo->opMode == CMK_DIRECT_API) {
    // ncpyOpInfo is not freed
    newNcpyOpInfo = ncpyOpInfo;

    packetKey = METADATAFIELD(newNcpyOpInfo)->key;

  } else if(ncpyOpInfo->opMode == CMK_EM_API) {

    // ncpyOpInfo is a part of the received message and can be freed before this send completes
    // for that reason, it is copied into a new message
    newNcpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyOpInfo->ncpyOpInfoSize);

    memcpy(newNcpyOpInfo, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize);

    newNcpyOpInfo->freeMe =  CMK_FREE_NCPYOPINFO; // Since this is a copy of ncpyOpInfo, it can be freed


    // Register the small message in order to send it to the other side
    packetKey = ibv_reg_mr(context->pd, (void *)newNcpyOpInfo, newNcpyOpInfo->ncpyOpInfoSize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    if (!packetKey) {
      CmiAbort("Memory Registration Failed in LrtsInvokeRemoteDeregAckHandler!\n");
    }
  } else {

    CmiAbort("Verbs: LrtsInvokeRemoteDeregAckHandler - ncpyOpInfo->opMode is not valid for dereg\n");
  }

  packet->size = newNcpyOpInfo->ncpyOpInfoSize;
  packet->buf  = (char *)newNcpyOpInfo;
  packet->header.code = INFIRDMA_DIRECT_DEREG_AND_ACK;
  packet->ogm  = NULL;

  OtherNode node = &nodes[CmiNodeOf(pe)];
  EnqueuePacket(node, packet, newNcpyOpInfo->ncpyOpInfoSize, packetKey);
}
