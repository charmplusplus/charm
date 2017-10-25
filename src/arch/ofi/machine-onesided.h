/* Support for the Ncpy Entry Method API */
#ifndef OFI_MACHINE_ONESIDED_H
#define OFI_MACHINE_ONESIDED_H

// Structure for sender side machine specific information for a buffer
typedef struct _cmi_ofi_rdma_op {
  int         len;
  uint64_t    key;
  uint64_t    mr;
  int         nodeNo;
  void        *ref;
  const void  *buf;
} CmiOfiRdmaSendOp_t;

// Structure for sender side machine specific information for multiple buffers
typedef struct _cmi_ofi_rdma{
  int            numOps;
  CmiOfiRdmaSendOp_t rdmaOp[0];
} CmiOfiRdmaSend_t;

// Structure for representing the acknowledgement information for a buffer
// This sturcture variable is sent by the receiver side on completion of an RDMA operation
typedef struct _ofi_rdma_ack {
  void     *src_ref;
  uint64_t src_mr;
} OfiRdmaAck_t;

// Structure for sender side machine specific information for a buffer
typedef struct _cmi_ofi_rdma_recv_op {
  int           len;
  uint64_t      src_key;
  int           src_nodeNo;
  const void    *src_buf;
  OfiRdmaAck_t    ack;
  int           completion_count;
  int           opIndex;
  const void    *buf;
  struct        fid_mr *mr;
} CmiOfiRdmaRecvOp_t;

// Structure for receiver side machine specific information for multiple buffers
typedef struct _cmi_ofi_rdma_recv {
  int numOps;
  int comOps;
  void *msg;
  CmiOfiRdmaRecvOp_t rdmaOp[0];
} CmiOfiRdmaRecv_t;

/* Sender Side Functions */
int LrtsGetRdmaOpInfoSize(){
  return sizeof(CmiOfiRdmaSendOp_t);
}

int LrtsGetRdmaGenInfoSize(){
  return sizeof(CmiOfiRdmaSend_t);
}

int LrtsGetRdmaInfoSize(int numOps){
  return sizeof(CmiOfiRdmaSend_t) + numOps * sizeof(CmiOfiRdmaSendOp_t);
}

void LrtsSetRdmaInfo(void *dest, int destPE, int numOps){
  CmiOfiRdmaSend_t *rdma = (CmiOfiRdmaSend_t*)dest;
  rdma->numOps = numOps;
}

void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  struct fid_mr *mr;
  uint64_t requested_key = 0;
  int ret;

  /* Register the source buffer */
  if(FI_MR_SCALABLE == context.mr_mode) {
    requested_key = __sync_fetch_and_add(&(context.mr_counter), 1);
  }
  ret = fi_mr_reg(context.domain,
                  ptr,
                  size,
                  MR_ACCESS_PERMISSIONS,
                  0ULL,
                  requested_key,
                  0ULL,
                  &mr,
                  NULL);

  if (ret) {
    CmiAbort("LrtsSetRdmaInfo: fi_mr_reg failed!\n");
  }

  CmiOfiRdmaSendOp_t *rdmaSendOpInfo = (CmiOfiRdmaSendOp_t *)dest;
  rdmaSendOpInfo->nodeNo  = CmiMyNodeGlobal();
  rdmaSendOpInfo->buf     = ptr;
  rdmaSendOpInfo->len     = size;
  rdmaSendOpInfo->key     = fi_mr_key(mr);
  rdmaSendOpInfo->mr      = (uint64_t)mr;
  rdmaSendOpInfo->ref     = ack;
}

/* Receiver Side Functions */
int LrtsGetRdmaOpRecvInfoSize(){
  return sizeof(CmiOfiRdmaRecvOp_t);
}

int LrtsGetRdmaGenRecvInfoSize(){
  return sizeof(CmiOfiRdmaRecv_t);
}

int LrtsGetRdmaRecvInfoSize(int numOps){
  return sizeof(CmiOfiRdmaRecv_t) + numOps * sizeof(CmiOfiRdmaRecvOp_t);
}

void LrtsSetRdmaRecvInfo(void *rdmaRecv, int numOps, void *msg, void *rdmaSend, int msgSize){
  CmiOfiRdmaRecv_t *rdmaRecvInfo = (CmiOfiRdmaRecv_t *)rdmaRecv;
  CmiOfiRdmaSend_t *rdmaSendInfo = (CmiOfiRdmaSend_t *)rdmaSend;

  rdmaRecvInfo->numOps = numOps;
  rdmaRecvInfo->comOps = 0;
  rdmaRecvInfo->msg = msg;
}

void LrtsSetRdmaRecvOpInfo(void *rdmaRecvOp, void *buffer, void *src_ref, int size, int opIndex, void *rdmaSend){
  int ret, len;
  struct fid_mr *mr = NULL;
  CmiOfiRdmaRecvOp_t *rdmaRecvOpInfo = (CmiOfiRdmaRecvOp_t *)rdmaRecvOp;
  CmiOfiRdmaSend_t *rdmaSendInfo = (CmiOfiRdmaSend_t *)rdmaSend;
  CmiOfiRdmaSendOp_t rdmaSendOpInfo = rdmaSendInfo->rdmaOp[opIndex];

  len = rdmaSendOpInfo.len;

  rdmaRecvOpInfo->len              = len;
  rdmaRecvOpInfo->src_key          = rdmaSendOpInfo.key;
  rdmaRecvOpInfo->src_nodeNo       = rdmaSendOpInfo.nodeNo;
  rdmaRecvOpInfo->src_buf          = rdmaSendOpInfo.buf;
  rdmaRecvOpInfo->ack.src_ref      = rdmaSendOpInfo.ref;
  rdmaRecvOpInfo->ack.src_mr       = rdmaSendOpInfo.mr;
  rdmaRecvOpInfo->completion_count = 0;
  rdmaRecvOpInfo->opIndex          = opIndex;
  rdmaRecvOpInfo->buf              = buffer;

  if (FI_MR_BASIC == context.mr_mode) {
    /* Register local MR to be read into */
    ret = fi_mr_reg(context.domain,
                    buffer,
                    len,
                    MR_ACCESS_PERMISSIONS,
                    0ULL,
                    0ULL,
                    0ULL,
                    &mr,
                    NULL);

    if (ret) {
      CmiAbort("LrtsSetRdmaRecvOpInfo: fi_mr_reg failed!\n");
    }
  }
  rdmaRecvOpInfo->mr            = mr;
}


inline void ofi_onesided_all_ops_done(char *msg);
inline void process_onesided_completion_ack(struct fi_cq_tagged_entry *e, OFIRequest *req);
#endif /* OFI_MACHINE_ONESIDED_H */
