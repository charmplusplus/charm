#ifndef MACHINE_ONESIDED_H_
#define MACHINE_ONESIDED_H_

typedef struct _cmi_gni_ack_op {
  CmiRdmaAck * ack;
  gni_mem_handle_t mem_hndl;
} CmiGNIAckOp_t;

typedef struct _cmi_gni_rzv_rdma_op {
  int size;
  uint64_t remote_addr;
  gni_mem_handle_t mem_hndl;
} CmiGNIRzvRdmaOp_t;

typedef struct _cmi_gni_rzv_rdma {
  int numOps;
  int srcPE;
  CmiGNIRzvRdmaOp_t rdmaOp[0];
} CmiGNIRzvRdma_t;

typedef struct _cmi_gni_rzv_rdma_recv_op {
  int size;
  uint64_t remote_addr;
  uint64_t local_addr;
  gni_mem_handle_t local_mem_hndl;
  gni_mem_handle_t remote_mem_hndl;
  int opIndex;
  void * src_info;
} CmiGNIRzvRdmaRecvOp_t;


typedef struct _cmi_gni_rzv_rdma_recv {
  int numOps;
  int comOps;
  int destNode;
  int srcPE;
  void* msg;
  CmiGNIRzvRdmaRecvOp_t rdmaOp[0];
} CmiGNIRzvRdmaRecv_t;

void _initOnesided();

void  rdma_sendAck (CmiGNIRzvRdmaRecvOp_t* recvOpInfo, int src_pe);
void  rdma_sendMsgForPutCompletion (CmiGNIRzvRdmaRecv_t* recvInfo, int dest_pe);

void LrtsIssueRgets(void *recv, int pe);
void LrtsIssueRputs(void *recv, int pe);

int LrtsGetRdmaOpInfoSize(){
  return sizeof(CmiGNIRzvRdmaOp_t);
}
int LrtsGetRdmaGenInfoSize(){
  return sizeof(CmiGNIRzvRdma_t);
}
int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiGNIRzvRdma_t) + numOps * sizeof(CmiGNIRzvRdmaOp_t);
}

int LrtsGetRdmaOpRecvInfoSize(){
  return sizeof(CmiGNIRzvRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(){
  return sizeof(CmiGNIRzvRdmaRecv_t);
}

int LrtsGetRdmaRecvInfoSize(int numOps){
  return sizeof(CmiGNIRzvRdmaRecv_t) + numOps * sizeof(CmiGNIRzvRdmaRecvOp_t);
}

void LrtsSetRdmaRecvInfo(void *rdmaRecv, int numOps, void *msg, void *rdmaSend, int msgSize){
  CmiGNIRzvRdmaRecv_t *rdmaRecvInfo = (CmiGNIRzvRdmaRecv_t *)rdmaRecv;
  CmiGNIRzvRdma_t *rdmaSendInfo = (CmiGNIRzvRdma_t *)rdmaSend;

  rdmaRecvInfo->srcPE = rdmaSendInfo->srcPE;
  rdmaRecvInfo->destNode = CmiMyNode();
  rdmaRecvInfo->numOps = numOps;
  rdmaRecvInfo->comOps = 0;
  rdmaRecvInfo->msg = msg;
}

void LrtsSetRdmaRecvOpInfo(void *rdmaRecvOp, void *buffer, void *src_ref, int size, int opIndex, void *rdmaSend){
  CmiGNIRzvRdmaRecvOp_t *rdmaRecvOpInfo = (CmiGNIRzvRdmaRecvOp_t *)rdmaRecvOp;
  CmiGNIRzvRdma_t *rdmaSendInfo = (CmiGNIRzvRdma_t *)rdmaSend;

  rdmaRecvOpInfo->src_info = src_ref;
  rdmaRecvOpInfo->size = rdmaSendInfo->rdmaOp[opIndex].size;
  rdmaRecvOpInfo->opIndex = opIndex;
  rdmaRecvOpInfo->local_addr = (uint64_t)buffer;

  rdmaRecvOpInfo->remote_addr = rdmaSendInfo->rdmaOp[opIndex].remote_addr;
  rdmaRecvOpInfo->remote_mem_hndl = rdmaSendInfo->rdmaOp[opIndex].mem_hndl;
}

void LrtsSetRdmaInfo(void *dest, int destPE, int numOps){
  CmiGNIRzvRdma_t *rdma = (CmiGNIRzvRdma_t*)dest;
  rdma->srcPE = CmiMyPe();
  rdma->numOps = numOps;
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  gni_mem_handle_t mem_hndl;
  gni_return_t status = GNI_RC_SUCCESS;

  status = GNI_MemRegister(nic_hndl, (uint64_t)ptr,  (uint64_t)size, NULL,  GNI_MEM_READ_ONLY, -1, &mem_hndl);
  GNI_RC_CHECK("Error! Exceeded Allowed Pinned Memory Limit! GNI_MemRegister on Sender Buffer (source) Failed before sending metadata message", status);

  CmiGNIRzvRdmaOp_t *rdmaOp = (CmiGNIRzvRdmaOp_t *)dest;
  rdmaOp->remote_addr = (uint64_t)ptr;
  rdmaOp->size = size;
  rdmaOp->mem_hndl = mem_hndl;
}

void PumpOneSidedRDMATransactions(gni_cq_handle_t rdma_cq, CmiNodeLock rdma_cq_lock);

#endif /* end if for MACHINE_ONESIDED_H_ */
