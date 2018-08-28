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

int LrtsGetRdmaOpInfoSize(){
  return sizeof(CmiVerbsRdmaOp_t);
}

int LrtsGetRdmaGenInfoSize(){
  return sizeof(CmiVerbsRdma_t);
}

int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiVerbsRdma_t) + numOps * sizeof(CmiVerbsRdmaOp_t);
}

int LrtsGetRdmaOpRecvInfoSize(){
  return sizeof(CmiVerbsRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(){
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

  mr = ibv_reg_mr(context->pd, (void *)ptr, size, IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!mr) {
    MACHSTATE(3, "ibv_reg_mr() failed\n");
  }
  rdmaOp->mr = mr;
  rdmaOp->key = mr->rkey;
}

#endif /* VERBS_MACHINE_ONESIDED_H */

typedef struct _cmi_verbs_rzv_rdma_pointer {
  struct ibv_mr *mr;
  uint32_t key;
}CmiVerbsRdmaPtr_t;

/* Compiler checks to ensure that CMK_NOCOPY_DIRECT_BYTES in conv-common.h
 * is set to sizeof(CmiVerbsRdmaPtr_t). CMK_NOCOPY_DIRECT_BYTES is used in
 * ckrdma.h to reserve bytes for source or destination metadata info.           */
#define DUMB_STATIC_ASSERT(test) typedef char sizeCheckAssertion[(!!(test))*2-1]

/* Machine specific metadata information required to register a buffer and perform
 * an RDMA operation with a remote buffer. This metadata information is used to perform
 * registration and a PUT operation when the remote buffer wants to perform a GET with an
 * unregistered buffer. Similary, the metadata information is used to perform registration
 * and a GET operation when the remote buffer wants to perform a PUT with an unregistered
 * buffer.*/
typedef struct _cmi_verbs_rdma_reverse_op {
  const void *destAddr;
  int destPe;
  int destMode;
  const void *srcAddr;
  int srcPe;
  int srcMode;

  struct ibv_mr *rem_mr;
  uint32_t rem_key;
  int ackSize;
  int size;
} CmiVerbsRdmaReverseOp_t;

/* Check the value of CMK_NOCOPY_DIRECT_BYTES if the compiler reports an
 * error with the message "the size of an array must be greater than zero" */
DUMB_STATIC_ASSERT(sizeof(CmiVerbsRdmaPtr_t) == CMK_NOCOPY_DIRECT_BYTES);

// Function Declarations
// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);

void postRdma(
  uint64_t local_addr,
  uint32_t local_rkey,
  uint64_t remote_addr,
  uint32_t remote_rkey,
  int size,
  int peNum,
  uint64_t rdmaPacket,
  int opcode);

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
  int size);

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
  int size);

void registerDirectMemory(void *info, const void *addr, int size);

