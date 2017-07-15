#ifndef VERBS_MACHINE_ONESIDED_H
#define VERBS_MACHINE_ONESIDED_H

typedef struct _cmi_verbs_rdma_op {
  uint64_t remote_addr;
  int size;
  struct ibv_mr *mr;
  uint32_t key;
} CmiVerbsRdmaOp_t;

typedef struct _cmi_verbs_rdma{
  int numOps;
  int peNum;
  CmiVerbsRdmaOp_t rdmaOp[0];
} CmiVerbsRdma_t;

typedef struct _cmi_verbs_rdma_recv_op {
  struct ibv_mr *remote_mr;
  struct ibv_mr *local_mr;
  void *src_info;
  uint64_t remote_addr;
  uint64_t local_addr;
  int size;
  uint32_t key;
  int opIndex;
} CmiVerbsRdmaRecvOp_t;

typedef struct _cmi_verbs_rdma_recv {
  int numOps;
  int comOps;
  int peNum;
  void *msg;
  CmiVerbsRdmaRecvOp_t rdmaOp[0];
} CmiVerbsRdmaRecv_t;

void verbsOnesidedOpDone(CmiVerbsRdmaRecvOp_t *recvOpInfo);

void verbsOnesidedSendAck(int peNum, CmiVerbsRdmaRecvOp_t *recvOpInfo);

void verbsOnesidedReceivedAck(struct infiRdmaPacket *rdmaPacket);

int LrtsGetRdmaOpInfoSize(void){
  return sizeof(CmiVerbsRdmaOp_t);
}

int LrtsGetRdmaGenInfoSize(void){
  return sizeof(CmiVerbsRdma_t);
}

int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiVerbsRdma_t) + numOps * sizeof(CmiVerbsRdmaOp_t);
}

int LrtsGetRdmaOpRecvInfoSize(void){
  return sizeof(CmiVerbsRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(void){
  return sizeof(CmiVerbsRdmaRecv_t);
}

int LrtsGetRdmaRecvInfoSize(int numOps){
  return sizeof(CmiVerbsRdmaRecv_t) + numOps * sizeof(CmiVerbsRdmaRecvOp_t);
}

void LrtsSetRdmaRecvInfo(void *rdmaRecv, int numOps, void *msg, void *rdmaSend, int msgSize){
  CmiVerbsRdmaRecv_t *rdmaRecvInfo = (CmiVerbsRdmaRecv_t *)rdmaRecv;
  CmiVerbsRdma_t *rdmaSendInfo = (CmiVerbsRdma_t *)rdmaSend;

  rdmaRecvInfo->numOps = numOps;
  rdmaRecvInfo->comOps = 0;
  rdmaRecvInfo->peNum = rdmaSendInfo->peNum;
  rdmaRecvInfo->msg = msg;
}

void LrtsSetRdmaRecvOpInfo(void *rdmaRecvOp, void *buffer, void *src_ref, int size, int opIndex, void *rdmaSend){
  CmiVerbsRdmaRecvOp_t *rdmaRecvOpInfo = (CmiVerbsRdmaRecvOp_t *)rdmaRecvOp;
  CmiVerbsRdma_t *rdmaSendInfo = (CmiVerbsRdma_t *)rdmaSend;

  rdmaRecvOpInfo->remote_mr = rdmaSendInfo->rdmaOp[opIndex].mr;
  rdmaRecvOpInfo->src_info = src_ref;
  rdmaRecvOpInfo->remote_addr = rdmaSendInfo->rdmaOp[opIndex].remote_addr;
  rdmaRecvOpInfo->local_addr = (uint64_t)buffer;
  rdmaRecvOpInfo->size = size;
  rdmaRecvOpInfo->key = rdmaSendInfo->rdmaOp[opIndex].key;
  rdmaRecvOpInfo->opIndex = opIndex;
}

void LrtsSetRdmaInfo(void *dest, int destPE, int numOps){
  CmiVerbsRdma_t *rdma = (CmiVerbsRdma_t*)dest;
  rdma->numOps = numOps;
  rdma->peNum = CmiMyPe();
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  struct ibv_mr *mr;

  CmiVerbsRdmaOp_t *rdmaOp = (CmiVerbsRdmaOp_t *)dest;
  rdmaOp->remote_addr = (uint64_t)ptr;
  rdmaOp->size = size;

  mr = ibv_reg_mr(context->pd, ptr, size, IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!mr) {
    MACHSTATE(3, "ibv_reg_mr() failed\n");
  }
  rdmaOp->mr = mr;
  rdmaOp->key = mr->rkey;
}

#endif /* VERBS_MACHINE_ONESIDED_H */
