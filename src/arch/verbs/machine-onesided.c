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

void postRdma(uint64_t local_addr, uint32_t local_rkey, uint64_t remote_addr, uint32_t remote_rkey, int size, int peNum, uint64_t rdmaPacket, int opcode) {

  CmiAssert(opcode == IBV_WR_RDMA_WRITE || opcode == IBV_WR_RDMA_READ);

  struct ibv_sge list = {
    .addr = (uintptr_t)local_addr,
    .length = size,
    .lkey = local_rkey
  };

  struct ibv_send_wr *bad_wr;
  struct ibv_send_wr wr = {
    .wr_id = rdmaPacket,
    .sg_list = &list,
    .num_sge = 1,
    .opcode = opcode,
    .send_flags = IBV_SEND_SIGNALED,
    .wr.rdma = {
      .remote_addr = remote_addr,
      .rkey = remote_rkey
    }
  };

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

  struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
  rdmaPacket->type = INFI_ONESIDED_DIRECT;
  rdmaPacket->localBuffer = ref;

  CmiVerbsRdmaPtr_t *tgt_info = (CmiVerbsRdmaPtr_t *)tgtInfo;
  CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)srcInfo;

  postRdma((uint64_t)tgtAddr,
          tgt_info->key,
          (uint64_t)srcAddr,
          src_info->key,
          size,
          srcPe,
          (uint64_t)rdmaPacket,
          IBV_WR_RDMA_WRITE);
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

  struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
  rdmaPacket->type = INFI_ONESIDED_DIRECT;
  rdmaPacket->localBuffer = ref;

  CmiVerbsRdmaPtr_t *src_info = (CmiVerbsRdmaPtr_t *)srcInfo;
  CmiVerbsRdmaPtr_t *tgt_info = (CmiVerbsRdmaPtr_t *)tgtInfo;

  postRdma((uint64_t)srcAddr,
          src_info->key,
          (uint64_t)tgtAddr,
          tgt_info->key,
          size,
          tgtPe,
          (uint64_t)rdmaPacket,
          IBV_WR_RDMA_WRITE);
}

// Method invoked to deregister target memory handle
void LrtsReleaseTargetResource(void *info, int pe){
  CmiVerbsRdmaPtr_t *rdmaTgt = (CmiVerbsRdmaPtr_t *)info;
  if (ibv_dereg_mr(rdmaTgt->mr)) {
    CmiAbort("ibv_dereg_mr() failed at LrtsReleaseTargetResource\n");
  }
}

// Method invoked to deregister source memory handle
void LrtsReleaseSourceResource(void *info, int pe){
  CmiVerbsRdmaPtr_t *rdmaSrc = (CmiVerbsRdmaPtr_t *)info;
  if (ibv_dereg_mr(rdmaSrc->mr)) {
    CmiAbort("ibv_dereg_mr() failed at LrtsReleaseSourceResource\n");
  }
}
