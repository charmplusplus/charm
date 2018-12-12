#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

/*Function Pointer to Acknowledgement Handler*/
typedef void (*RdmaAckHandlerFn)(void *token);

/* Support for Nocopy Direct API */
typedef struct _cmi_common_rdma_info {
#if CMK_USE_CMA
  pid_t pid;
#elif defined _MSC_VER
  char empty;
#endif
} CmiCommonRdmaInfo_t;

/* Set the generic converse/LRTS information */
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size) {
#if CMK_USE_CMA
  CmiCommonRdmaInfo_t *cmmInfo = (CmiCommonRdmaInfo_t *)info;
  cmmInfo->pid = getpid();
#endif
}

int CmiGetRdmaCommonInfoSize() {
  return sizeof(CmiCommonRdmaInfo_t);
}

#if CMK_USE_CMA
#include <sys/uio.h> // for struct iovec
extern int cma_works;
int readShmCma(pid_t, struct iovec *, struct iovec *, int, size_t);
int writeShmCma(pid_t, struct iovec *, struct iovec *, int, size_t);

// These methods are also used by the generic layer implementation of the Direct API
void CmiIssueRgetUsingCMA(
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  const void* destAddr,
  void *destInfo,
  int destPe,
  int size) {

  // Use SHM transport for a PE on the same host
  struct iovec local, remote;
  // local memory address
  local.iov_base = (void *)destAddr;
  local.iov_len  = size;

  // remote memory address
  remote.iov_base = (void *)srcAddr;
  remote.iov_len  = size;

  // get remote process id
  CmiCommonRdmaInfo_t *remoteCommInfo = (CmiCommonRdmaInfo_t *)srcInfo;
  pid_t pid = remoteCommInfo->pid;
  readShmCma(pid, &local, &remote, 1, size);
}

void CmiIssueRputUsingCMA(
  const void* destAddr,
  void *destInfo,
  int destPe,
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  int size) {

  // Use SHM transport for a PE on the same host
  struct iovec local, remote;
  // local memory address
  local.iov_base = (void *)srcAddr;
  local.iov_len  = size;

  // remote memory address
  remote.iov_base = (void *)destAddr;
  remote.iov_len  = size;

  // get remote process id
  CmiCommonRdmaInfo_t *remoteCommInfo = (CmiCommonRdmaInfo_t *)destInfo;
  pid_t pid = remoteCommInfo->pid;
  writeShmCma(pid, &local, &remote, 1, size);
}
#endif

#if CMK_ONESIDED_IMPL

// Function Pointer to the acknowledement handler function for the Direct API
RdmaAckHandlerFn ncpyDirectAckHandlerFn;

typedef struct _cmi_rdma_direct_ack {
  const void *srcAddr;
  int srcPe;
  const void *destAddr;
  int destPe;
  int ackSize;
} CmiRdmaDirectAck;

/* Support for Nocopy Direct API */
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);
void LrtsSetRdmaNcpyAck(RdmaAckHandlerFn fn);
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo);

void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo);

void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode);

/* Set the machine specific information for a nocopy pointer */
void CmiSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){
  LrtsSetRdmaBufferInfo(info, ptr, size, mode);
}

void CmiInvokeNcpyAck(void *ack) {
  ncpyDirectAckHandlerFn(ack);
}

/* Set the ack handler function used in the Direct API */
void CmiSetDirectNcpyAckHandler(RdmaAckHandlerFn fn){
  ncpyDirectAckHandlerFn = fn;
}

/* Perform an RDMA Get operation into the local destination address from the remote source address*/
void CmiIssueRget(NcpyOperationInfo *ncpyOpInfo) {
  // Use network RDMA for a PE on a remote host
  LrtsIssueRget(ncpyOpInfo);
}

/* Perform an RDMA Put operation into the remote destination address from the local source address */
void CmiIssueRput(NcpyOperationInfo *ncpyOpInfo) {
  // Use network RDMA for a PE on a remote host
  LrtsIssueRput(ncpyOpInfo);
}

/* De-register registered memory for pointer */
void CmiDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
  LrtsDeregisterMem(ptr, info, pe, mode);
}

#endif /*End of CMK_ONESIDED_IMPL */
#endif
