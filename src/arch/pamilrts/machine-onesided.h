void _initOnesided( pami_context_t *contexts, int nc);

typedef struct _cmi_pami_rzv_rdma_op {
  int offset;
  int size;
} CmiPAMIRzvRdmaOp_t;

typedef struct _cmi_pami_rzv_rdma{
  int numOps;
  size_t dstContext;
  pami_memregion_t    mregion;
  CmiPAMIRzvRdmaOp_t rdmaOp[0];
} CmiPAMIRzvRdma_t;

typedef struct _cmi_pami_rzv_rdma_recv_op {
  void           * buffer;
  void           * src_info;
  int              offset;
  int              size;
  int              opIndex;
} CmiPAMIRzvRdmaRecvOp_t;

typedef struct _cmi_pami_rzv_rdma_recv {
  int           src_ep;
  int           numOps;
  int           comOps;
  void*         msg;
  size_t           dstContext;
  pami_memregion_t    mregion;
  CmiPAMIRzvRdmaRecvOp_t rdmaOp[0];
}CmiPAMIRzvRdmaRecv_t;

void ack_rdma_pkt_dispatch (pami_context_t       context,
    void               * clientdata,
    const void         * header_addr,
    size_t               header_size,
    const void         * pipe_addr,
    size_t               pipe_size,
    pami_endpoint_t      origin,
    pami_recv_t         * recv);

void  rdma_sendAck (
    pami_context_t      context,
    CmiPAMIRzvRdmaRecvOp_t* recvOpInfo,
    int src_ep);

void rzv_rdma_recv_done   (
    pami_context_t     ctxt,
    void             * clientdata,
    pami_result_t      result);

int LrtsGetRdmaOpInfoSize(){
  return sizeof(CmiPAMIRzvRdmaOp_t);
}
int LrtsGetRdmaGenInfoSize(){
  return sizeof(CmiPAMIRzvRdma_t);
}
int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiPAMIRzvRdma_t) + numOps * sizeof(CmiPAMIRzvRdmaOp_t);
}

int LrtsGetRdmaOpRecvInfoSize(){
  return sizeof(CmiPAMIRzvRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(){
  return sizeof(CmiPAMIRzvRdmaRecv_t);
}

int LrtsGetRdmaRecvInfoSize(int numOps){
  return sizeof(CmiPAMIRzvRdmaRecv_t) + numOps * sizeof(CmiPAMIRzvRdmaRecvOp_t);
}

void LrtsSetRdmaRecvInfo(void *rdmaRecv, int numOps, void *msg, void *rdmaSend, int msgSize){

  CmiPAMIRzvRdmaRecv_t *rdmaRecvInfo = (CmiPAMIRzvRdmaRecv_t *)rdmaRecv;
  CmiPAMIRzvRdma_t *rdmaSendInfo = (CmiPAMIRzvRdma_t *)rdmaSend;

  rdmaRecvInfo->numOps = numOps;
  rdmaRecvInfo->comOps = 0;
  rdmaRecvInfo->msg = msg;

  rdmaRecvInfo->dstContext = rdmaSendInfo->dstContext;
  memcpy(&rdmaRecvInfo->mregion, &rdmaSendInfo->mregion, sizeof(pami_memregion_t));
}

void LrtsSetRdmaRecvOpInfo(void *rdmaRecvOp, void *buffer, void *src_ref, int size, int opIndex, void *rdmaSend){
  CmiPAMIRzvRdmaRecvOp_t *rdmaRecvOpInfo = (CmiPAMIRzvRdmaRecvOp_t *)rdmaRecvOp;
  CmiPAMIRzvRdma_t *rdmaSendInfo = (CmiPAMIRzvRdma_t *)rdmaSend;

  rdmaRecvOpInfo->buffer = buffer;
  rdmaRecvOpInfo->src_info = src_ref;
  rdmaRecvOpInfo->size = size;
  rdmaRecvOpInfo->opIndex = opIndex;

  rdmaRecvOpInfo->offset = rdmaSendInfo->rdmaOp[opIndex].offset;
}


void LrtsSetRdmaInfo(void *dest, int destPE, int numOps){

  CmiPAMIRzvRdma_t *rdma = (CmiPAMIRzvRdma_t*)dest;
  int rank = CmiRankOf(destPE);

#if CMK_PAMI_MULTI_CONTEXT &&  CMK_NODE_QUEUE_AVAILABLE
  size_t dst_context = (rank != DGRAM_NODEMESSAGE) ? (rank>>LTPS) : (rand_r(&r_seed) % cmi_pami_numcontexts);
#else
  size_t dst_context = 0;
#endif

  rdma->dstContext = dst_context;
  memcpy(&rdma->mregion, &cmi_pami_memregion[0].mregion, sizeof(pami_memregion_t));
  rdma->numOps = numOps;
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  CmiPAMIRzvRdmaOp_t *rdmaOp = (CmiPAMIRzvRdmaOp_t *)dest;
  rdmaOp->offset = (size_t)(ptr) - (size_t)cmi_pami_memregion[0].baseVA;
  rdmaOp->size = size;
}
