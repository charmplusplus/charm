/*
 * included in machine-ibverbs.C
 * Jaemin Choi
 */

void verbsOnesidedAllOpsDone(char *msg) {
  int sndlen = ((CmiMsgHeaderBasic *) msg)->size;

  handleOneRecvedMsg(sndlen, msg);
}

/* function called on completion of the rdma operation */
void verbsOnesidedOpDone(CmiVerbsRdmaRecvOp_t *recvOpInfo) {
  CmiVerbsRdmaRecv_t *recvInfo = (CmiVerbsRdmaRecv_t *)(
      ((char *)recvOpInfo) - recvOpInfo->opIndex * sizeof(CmiVerbsRdmaRecvOp_t)
    - sizeof(CmiVerbsRdmaRecv_t));

  if (ibv_dereg_mr(recvOpInfo->local_mr)) {
    MACHSTATE(3, "ibv_dereg_mr() failed\n");
  }

  verbsOnesidedSendAck(recvInfo->peNum, recvOpInfo);
  recvInfo->comOps++;
  if (recvInfo->comOps == recvInfo->numOps) {
    verbsOnesidedAllOpsDone((char *)recvInfo->msg);
  }
}

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

/* function to perform RDMA read operation */
void verbsOnesidedPostRdmaRead(int peNum, CmiVerbsRdmaRecvOp_t *recvOpInfo) {
  struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)malloc(sizeof(struct infiRdmaPacket));
  rdmaPacket->type = INFI_ONESIDED;
  rdmaPacket->localBuffer = recvOpInfo;

#if CMK_IBVERBS_TOKENS_FLOW
  context->tokensLeft--;
#if CMK_IBVERBS_STATS
  if(context->tokensLeft < minTokensLeft)
    minTokensLeft = context->tokensLeft;
#endif
#endif

  struct ibv_mr *mr;
  mr = ibv_reg_mr(context->pd, (void *)recvOpInfo->local_addr, recvOpInfo->size, IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    MACHSTATE(3, "ibv_reg_mr() failed\n");
  }
  recvOpInfo->local_mr = mr;

  postRdma(recvOpInfo->local_addr,
          mr->lkey,
          recvOpInfo->remote_addr,
          recvOpInfo->key,
          recvOpInfo->size,
          peNum,
          (uint64_t)rdmaPacket,
          IBV_WR_RDMA_READ);
}

void LrtsIssueRgets(void *recv, int pe) {
  CmiVerbsRdmaRecv_t* recvInfo = (CmiVerbsRdmaRecv_t*)recv;
  int peNum = recvInfo->peNum;
  int i;

  for (i = 0; i < recvInfo->numOps; i++)
    verbsOnesidedPostRdmaRead(peNum, &recvInfo->rdmaOp[i]);
}

/* function to send the acknowledgement to the sender */
void verbsOnesidedSendAck(int peNum, CmiVerbsRdmaRecvOp_t *recvOpInfo) {
  struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)CmiAlloc(sizeof(struct infiRdmaPacket));
  rdmaPacket->fromNodeNo = CmiNodeOf(peNum);
  rdmaPacket->type = INFI_ONESIDED;
  rdmaPacket->keyPtr = recvOpInfo->remote_mr;
  rdmaPacket->localBuffer = recvOpInfo->src_info;

  EnqueueRdmaAck(rdmaPacket);
  CmiFree(rdmaPacket);
}

/* function called on the sender on receiving acknowledgement
 * from the receiver to signal the completion of the rdma operation */
void verbsOnesidedReceivedAck(struct infiRdmaPacket *rdmaPacket) {
  struct ibv_mr *mr = rdmaPacket->keyPtr;
  if (ibv_dereg_mr(mr)) {
    MACHSTATE(3, "ibv_dereg_mr() failed\n");
  }

  CmiRdmaAck *ack = (CmiRdmaAck *)rdmaPacket->localBuffer;
  ack->fnPtr(ack->token);

  //free callback structure, CmiRdmaAck allocated in CmiSetRdmaAck
  free(ack);
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

    struct ibv_mr *packetKey = METADATAFIELD(ncpyOpInfo)->key;
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
            ncpyOpInfo->srcSize,
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
            ncpyOpInfo->srcSize,
            ncpyOpInfo->destPe,
            (uint64_t)rdmaPacket,
            IBV_WR_RDMA_WRITE);
  }
}

// Method invoked to deregister a memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
  CmiVerbsRdmaPtr_t *rdmadest = (CmiVerbsRdmaPtr_t *)info;

  if (ibv_dereg_mr(rdmadest->mr)) {
    CmiAbort("ibv_dereg_mr() failed at LrtsDeregisterMem\n");
  }
}
