/*
 * included in machine-ibverbs.c
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
    verbsOnesidedAllOpsDone(recvInfo->msg);
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
  wr.opcode = opcode;
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
  struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
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

  CmiRdmaAck *ack = rdmaPacket->localBuffer;
  ack->fnPtr(ack->token);

  //free callback structure, CmiRdmaAck allocated in CmiSetRdmaAck
  free(ack);
}

/* Support for Nocopy Direct API */

struct ibv_mr* registerDirectMemory(const void *addr, int size) {
  struct ibv_mr *mr = ibv_reg_mr(context->pd, (void *)addr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    CmiAbort("Memory Registration at Destination Failed inside registerDirectMemory!\n");
  }
  return mr;
}

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  int size) {

  CmiAssert(srcAckSize == destAckSize);

  // Register local buffer if it is not registered
  if(*destMode == CMK_BUFFER_UNREG) {
    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)destInfo;
    dest_info->mr = registerDirectMemory(destAddr, size);
    dest_info->key = dest_info->mr->rkey;
    *destMode = CMK_BUFFER_REG;
  }

  if(*srcMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform PUT
    int mdMsgSize = sizeof(CmiVerbsRdmaReverseOp_t) + destAckSize + srcAckSize;
    CmiVerbsRdmaReverseOp_t *regAndPutMsg = (CmiVerbsRdmaReverseOp_t *)CmiAlloc(mdMsgSize);
    regAndPutMsg->destAddr = destAddr;
    regAndPutMsg->destPe   = destPe;
    regAndPutMsg->destMode = *destMode;
    regAndPutMsg->srcAddr  = srcAddr;
    regAndPutMsg->srcPe    = srcPe;
    regAndPutMsg->srcMode  = *srcMode;
    regAndPutMsg->rem_mr   = ((CmiVerbsRdmaPtr_t *)destInfo)->mr;
    regAndPutMsg->rem_key  = regAndPutMsg->rem_mr->rkey;
    regAndPutMsg->ackSize  = destAckSize;
    regAndPutMsg->size     = size;

    memcpy((char*)regAndPutMsg + sizeof(CmiVerbsRdmaReverseOp_t), destAck, destAckSize);
    memcpy((char*)regAndPutMsg + sizeof(CmiVerbsRdmaReverseOp_t) + srcAckSize, srcAck, srcAckSize);

    infiPacket packet;
    MallocInfiPacket(packet);

    packet->size = mdMsgSize;
    packet->buf  = (void *)regAndPutMsg;
    packet->header.code = INFIRDMA_DIRECT_REG_AND_PUT;
    packet->ogm  = NULL;

    struct ibv_mr *packetKey = METADATAFIELD(regAndPutMsg)->key;
    OtherNode node = &nodes[CmiNodeOf(srcPe)];
    EnqueuePacket(node, packet, mdMsgSize, packetKey);

  } else {
    void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, destAddr, destAck, destPe, srcAckSize);

    struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
    rdmaPacket->type = INFI_ONESIDED_DIRECT;
    rdmaPacket->localBuffer = ref;

    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)destInfo;
    CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)srcInfo;

    postRdma((uint64_t)destAddr,
            dest_info->key,
            (uint64_t)srcAddr,
            src_info->key,
            size,
            srcPe,
            (uint64_t)rdmaPacket,
            IBV_WR_RDMA_READ);
  }
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  int size) {

  CmiAssert(srcAckSize == destAckSize);

  // Register local buffer if it is not registered
  if(*srcMode == CMK_BUFFER_UNREG) {
    CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)srcInfo;
    src_info->mr = registerDirectMemory(srcAddr, size);
    src_info->key = src_info->mr->rkey;
    *srcMode = CMK_BUFFER_REG;
  }

  if(*destMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform GET
    int mdMsgSize = sizeof(CmiVerbsRdmaReverseOp_t) + destAckSize + srcAckSize;
    CmiVerbsRdmaReverseOp_t *regAndGetMsg = (CmiVerbsRdmaReverseOp_t *)CmiAlloc(mdMsgSize);
    regAndGetMsg->srcAddr  = srcAddr;
    regAndGetMsg->srcPe    = srcPe;
    regAndGetMsg->srcMode  = *srcMode;

    regAndGetMsg->destAddr = destAddr;
    regAndGetMsg->destPe   = destPe;
    regAndGetMsg->destMode = *destMode;

    regAndGetMsg->rem_mr   = ((CmiVerbsRdmaPtr_t *)srcInfo)->mr;
    regAndGetMsg->rem_key  = regAndGetMsg->rem_mr->rkey;
    regAndGetMsg->ackSize  = srcAckSize;
    regAndGetMsg->size     = size;

    memcpy((char*)regAndGetMsg + sizeof(CmiVerbsRdmaReverseOp_t), destAck, destAckSize);
    memcpy((char*)regAndGetMsg + sizeof(CmiVerbsRdmaReverseOp_t) + srcAckSize, srcAck, srcAckSize);

    infiPacket packet;
    MallocInfiPacket(packet);

    packet->size = mdMsgSize;
    packet->buf  = (void *)regAndGetMsg;
    packet->header.code = INFIRDMA_DIRECT_REG_AND_GET;
    packet->ogm  = NULL;

    struct ibv_mr *packetKey = METADATAFIELD(regAndGetMsg)->key;
    OtherNode node = &nodes[CmiNodeOf(destPe)];
    EnqueuePacket(node, packet, mdMsgSize, packetKey);

  } else {
    void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, destAddr, destAck, destPe, srcAckSize);

    struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
    rdmaPacket->type = INFI_ONESIDED_DIRECT;
    rdmaPacket->localBuffer = ref;

    CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)srcInfo;
    CmiVerbsRdmaPtr_t *dest_info = (CmiVerbsRdmaPtr_t *)destInfo;

    postRdma((uint64_t)srcAddr,
            src_info->key,
            (uint64_t)destAddr,
            dest_info->key,
            size,
            destPe,
            (uint64_t)rdmaPacket,
            IBV_WR_RDMA_WRITE);
  }
}

// Method invoked to deregister destination memory handle
void LrtsReleaseDestinationResource(const void *ptr, void *info, int pe, unsigned short int mode){
  if(mode == CMK_BUFFER_REG) {
    CmiVerbsRdmaPtr_t *rdmadest = (CmiVerbsRdmaPtr_t *)info;
    if (ibv_dereg_mr(rdmadest->mr)) {
      CmiAbort("ibv_dereg_mr() failed at LrtsReleaseDestinationResource\n");
    }
  }
}

// Method invoked to deregister source memory handle
void LrtsReleaseSourceResource(const void *ptr, void *info, int pe, unsigned short int mode){
  if(mode == CMK_BUFFER_REG) {
    CmiVerbsRdmaPtr_t *rdmaSrc = (CmiVerbsRdmaPtr_t *)info;
    if (ibv_dereg_mr(rdmaSrc->mr)) {
      CmiAbort("ibv_dereg_mr() failed at LrtsReleaseSourceResource\n");
    }
  }
}
