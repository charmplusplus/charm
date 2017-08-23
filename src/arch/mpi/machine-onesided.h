
typedef struct _cmi_mpi_rzv_rdma_op{
  int size;
  int tag;
} CmiMPIRzvRdmaOp_t;

typedef struct _cmi_mpi_rzv_rdma{
  int numOps;
  int srcRank;
  CmiMPIRzvRdmaOp_t rdmaOp[0];
} CmiMPIRzvRdma_t;

typedef struct _cmi_mpi_rzv_rdma_recv_op {
  void *buffer;
  void *src_info;
  int size;
  int tag;
  int hasCompleted;
  MPI_Request req;
} CmiMPIRzvRdmaRecvOp_t;

//Receiver's rdma buffer information is stored as a list to wait for completion of the recv requests
typedef struct _cmi_mpi_rzv_rdma_recv_list {
  int srcRank;
  int numOps;
  int msgLen;
  void *msg;
  struct _cmi_mpi_rzv_rdma_recv_list* next;
  CmiMPIRzvRdmaRecvOp_t rdmaOp[0];
} CmiMPIRzvRdmaRecvList_t;

//Used by MPI Send
typedef struct _cmi_mpi_rzv_rdma_opinfo {
  void *ack;
  int tag;
} CmiMPIRzvRdmaOpInfo_t;

//List of rdma receiver buffer information that is used for waiting for completion
CpvDeclare(CmiMPIRzvRdmaRecvList_t *, recvRdmaBuffers);
CpvDeclare(CmiMPIRzvRdmaRecvList_t *, endRdmaBuffer);
CpvDeclare(int, RdmaRecvQueueLen);

void MPIPostOneBuffer(const void *buffer, void *ref, int size, int pe, int tag, int type);

int getNewMPITag(){

  /* A local variable is used to avoid a race condition that can occur when
   * the global variable rdmaTag is updated by another thread after the first
   * thread releases the lock but before it sends the variable back */
  int newRdmaTag;
#if CMK_SMP
  CmiLock(rdmaTagLock);
#endif

  rdmaTag++;

  /* Reset generated tag when equal to the implementation dependent upper bound.
   * This condition also ensures correct resetting of the generated tag if tagUb is INT_MAX */
  if(rdmaTag == tagUb)
    rdmaTag = RDMA_BASE_TAG; //reseting can fail if previous tags are in use

  //copy the updated value into the local variable to ensure consistent a tag value
  newRdmaTag = rdmaTag;
#if CMK_SMP
  CmiUnlock(rdmaTagLock);
#endif
  return newRdmaTag;
}

int LrtsGetRdmaOpInfoSize(){
  return sizeof(CmiMPIRzvRdmaOp_t);
}

int LrtsGetRdmaGenInfoSize(){
  return sizeof(CmiMPIRzvRdma_t);
}

int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiMPIRzvRdma_t) + numOps * sizeof(CmiMPIRzvRdmaOp_t);
}

int LrtsGetRdmaOpRecvInfoSize(){
  return sizeof(CmiMPIRzvRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(){
  return sizeof(CmiMPIRzvRdmaRecvList_t);
}

int LrtsGetRdmaRecvInfoSize(int numOps){
  return sizeof(CmiMPIRzvRdmaRecvList_t) + numOps * sizeof(CmiMPIRzvRdmaRecvOp_t);
}

void LrtsSetRdmaRecvInfo(void *rdmaRecv, int numOps, void *msg, void *rdmaSend, int msgSize){
  CmiMPIRzvRdmaRecvList_t *rdmaRecvInfo = (CmiMPIRzvRdmaRecvList_t *)rdmaRecv;
  CmiMPIRzvRdma_t *rdmaSendInfo = (CmiMPIRzvRdma_t *)rdmaSend;

  rdmaRecvInfo->numOps = numOps;
  rdmaRecvInfo->srcRank = rdmaSendInfo->srcRank;
  rdmaRecvInfo->msg = msg;
  rdmaRecvInfo->next = 0;
  rdmaRecvInfo->msgLen = msgSize;
}

void LrtsSetRdmaRecvOpInfo(void *rdmaRecvOp, void *buffer, void *src_ref, int size, int opIndex, void *rdmaSend){
  CmiMPIRzvRdmaRecvOp_t *rdmaRecvOpInfo = (CmiMPIRzvRdmaRecvOp_t *)rdmaRecvOp;
  CmiMPIRzvRdma_t *rdmaSendInfo = (CmiMPIRzvRdma_t *)rdmaSend;

  rdmaRecvOpInfo->buffer = buffer;
  rdmaRecvOpInfo->size = size;
  rdmaRecvOpInfo->src_info = src_ref;

  rdmaRecvOpInfo->tag = rdmaSendInfo->rdmaOp[opIndex].tag;
  rdmaRecvOpInfo->hasCompleted = 0;
}

void LrtsSetRdmaInfo(void *dest, int destPE, int numOps){
  CmiMPIRzvRdma_t *rdma = (CmiMPIRzvRdma_t *)dest;
  rdma->numOps = numOps;
  /* srcRank is a global variable that stores the sender rank as we cannot
   * call MPI_Comm_rank from this thread as it is worker thread */
  rdma->srcRank = srcRank;
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  CmiMPIRzvRdmaOp_t *rdmaOp = (CmiMPIRzvRdmaOp_t *)dest;
  rdmaOp->size = size;

  //Generate a new tag to be used for the RDMA buffer
  rdmaOp->tag = getNewMPITag();

  CmiMPIRzvRdmaOpInfo_t *rdmaOpInfo = (CmiMPIRzvRdmaOpInfo_t *) malloc(sizeof(CmiMPIRzvRdmaOpInfo_t));
  rdmaOpInfo->ack = ack;
  rdmaOpInfo->tag = rdmaOp->tag;

  // Post the RDMA buffer with the generated tag using MPI_Isend. Post MPI_Isend directly for non-smp or through the comm thread for smp mode
  MPIPostOneBuffer(ptr, (void *)rdmaOpInfo, size, destPE, rdmaOp->tag, ONESIDED_BUFFER);
}

// Structure used for the Nocopy Direct API to request an MPI rank to post a buffer
typedef struct _cmi_mpi_rzv_rdma_post_info {
  void *buffer;
  int size;
  int tag;
  int ackSize;
  int srcPe;
  int tgtPe;
}CmiMPIRzvRdmaPostInfo_t;

// Structure used for the Nocopy Direct API for acknowledgement handling
typedef struct _cmi_mpi_rzv_rdma_ack_info {
  int pe;
  int tag;
}CmiMPIRzvRdmaAckInfo_t;

// Set the machine specific information for a nocopy source pointer
void LrtsSetRdmaSrcInfo(void *info, const void *ptr, int size){
}

// Set the machine specific information for a nocopy target pointer
void LrtsSetRdmaTgtInfo(void *info, const void *ptr, int size){
}
