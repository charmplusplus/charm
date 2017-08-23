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


/* Support for Nocopy Direct API */

/* Type of RDMA transaction
   This is used in handling each type of RDMA transaction appropriately on completion
 */
enum CMK_RDMA_TYPE {
  INDIRECT_SEND=1,
  DIRECT_SEND_RECV,
  DIRECT_SEND_RECV_UNALIGNED
};

// Machine specific information for a nocopy source pointer
typedef struct _cmi_gni_rzv_rdma_source {
  // memory handle for the source buffer
  gni_mem_handle_t mem_hndl;
} CmiGNIRzvRdmaSrc_t;

// Machine specific information for a nocopy target pointer
typedef struct _cmi_gni_rzv_rdma_target {
  // memory handle for the target buffer
  gni_mem_handle_t mem_hndl;
} CmiGNIRzvRdmaTgt_t;

/* Machine specific metadata information required for a PUT operation
 * This structure is used for an unaligned GET, which uses PUT underneath
 */
typedef struct _cmi_gni_rzv_rdma_put_op {
  gni_mem_handle_t tgt_mem_hndl;
  uint64_t tgt_addr;
  gni_mem_handle_t src_mem_hndl;
  uint64_t src_addr;
  int destPe;
  int size;
  uint64_t ref;
} CmiGNIRzvRdmaPutOp_t;

// Set the machine specific information for a nocopy source pointer
void LrtsSetRdmaSrcInfo(void *info, const void *ptr, int size){
  gni_mem_handle_t mem_hndl;
  gni_return_t status = GNI_RC_SUCCESS;
  status = GNI_MemRegister(nic_hndl, (uint64_t)ptr, (uint64_t)size, NULL, GNI_MEM_READ_ONLY, -1, &mem_hndl);
  GNI_RC_CHECK("Error! Source memory registration failed!", status);

  CmiGNIRzvRdmaSrc_t *rdmaSrc = (CmiGNIRzvRdmaSrc_t *)info;
  rdmaSrc->mem_hndl = mem_hndl;
}

// Set the machine specific information for a nocopy target pointer
void LrtsSetRdmaTgtInfo(void *info, const void *ptr, int size){
  gni_mem_handle_t mem_hndl;
  gni_return_t status = GNI_RC_SUCCESS;
  status = GNI_MemRegister(nic_hndl, (uint64_t)ptr, (uint64_t)size, NULL, GNI_MEM_READWRITE, -1, &mem_hndl);
  GNI_RC_CHECK("Error! Target memory registration failed!", status);
  CmiGNIRzvRdmaTgt_t *rdmaTgt = (CmiGNIRzvRdmaTgt_t *)info;
  rdmaTgt->mem_hndl = mem_hndl;
}

// Method performs RDMA operations
gni_return_t post_rdma(
  uint64_t remote_addr,
  gni_mem_handle_t remote_mem_hndl,
  uint64_t local_addr,
  gni_mem_handle_t local_mem_hndl,
  int length,
  uint64_t post_id,
  int destNode,
  int type,
  int mode);

// Method deregisters local and remote memory handles
void DeregisterMemhandle(gni_mem_handle_t mem_hndl, int pe);
#endif /* end if for MACHINE_ONESIDED_H_ */
