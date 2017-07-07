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

/* function to perform RDMA read operation */
void verbsOnesidedPostRdmaRead(int peNum, CmiVerbsRdmaRecvOp_t *recvOpInfo) {
  struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
  rdmaPacket->type = INFI_ONESIDED;
  rdmaPacket->localBuffer = recvOpInfo;

  uint64_t local_addr = recvOpInfo->local_addr;
  uint64_t remote_addr = recvOpInfo->remote_addr;
  int size = recvOpInfo->size;
  uint32_t rkey = recvOpInfo->key;

#if CMK_IBVERBS_TOKENS_FLOW
  context->tokensLeft--;
#if CMK_IBVERBS_STATS
  if(context->tokensLeft < minTokensLeft)
    minTokensLeft = context->tokensLeft;
#endif
#endif

  struct ibv_mr *mr;
  mr = ibv_reg_mr(context->pd, (void *)local_addr, size, IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    MACHSTATE(3, "ibv_reg_mr() failed\n");
  }
  recvOpInfo->local_mr = mr;

  struct ibv_sge list = {
    .addr = (uintptr_t)local_addr,
    .length = size,
    .lkey = mr->lkey
  };

  struct ibv_send_wr *bad_wr;
  struct ibv_send_wr wr = {
    .wr_id = (uint64_t)rdmaPacket,
    .sg_list = &list,
    .num_sge = 1,
    .opcode = IBV_WR_RDMA_READ,
    .send_flags = IBV_SEND_SIGNALED,
    .wr.rdma = {
      .remote_addr = remote_addr,
      .rkey = rkey
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
    sprintf(errMsg,"ibv_post_send failed in verbsOnesidedPostRdmaRead!! Try passing a larger value for maxTokens (%d) using argument +IBVMaxSendTokens\n",maxTokens);
    CmiAbort(errMsg);
  }
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
