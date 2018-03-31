#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

typedef void (*RdmaSingleAckHandlerFn)(void *cbPtr, int pe, const void *ptr);
/*Function Pointer to Acknowledgement Handler*/
typedef void (*RdmaAckHandlerFn)(void *token);

/*Acknowledgement constisting of handler and token*/
typedef struct _cmi_rdma_ack{
  // Function Pointer to Acknowledgment handler function for the Indirect API
  RdmaAckHandlerFn fnPtr;
  void *token;
} CmiRdmaAck;

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
/*Lrts Function declarations*/

/*Sender Functions*/
void LrtsSetRdmaInfo(void *dest, int destPE, int numOps);
void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE);
int LrtsGetRdmaOpInfoSize(void);
int LrtsGetRdmaGenInfoSize(void);
int LrtsGetRdmaInfoSize(int numOps);
void LrtsSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize);

/*Receiver Functions*/
void LrtsSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaRecv);
int LrtsGetRdmaOpRecvInfoSize(void);
int LrtsGetRdmaGenRecvInfoSize(void);
int LrtsGetRdmaRecvInfoSize(int numOps);
void LrtsIssueRgets(void *recv, int pe);
/* Converse Machine Interface Functions*/

/* Sender Side Functions */

/* Set the machine layer info generic to RDMA ops*/
void CmiSetRdmaInfo(void *dest, int destPE, int numOps){
  LrtsSetRdmaInfo(dest, destPE, numOps);
}

/* Set the machine layer info specific to RDMA op*/
void CmiSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  LrtsSetRdmaOpInfo(dest, ptr, size, ack, destPE);
}

/* Getter for size help upper layers allocate space for machine layer info
 * while allocating the message*/

/* Get the size occupied by the machine layer info specific to RDMA op*/
int CmiGetRdmaOpInfoSize(void){
  return LrtsGetRdmaOpInfoSize();
}

/* Get the size occupied by the macine layer info generic to RDMA ops*/
int CmiGetRdmaGenInfoSize(void){
  return LrtsGetRdmaGenInfoSize();
}

/* Get the total size occupied by the machine layer info (specific + generic)*/
int CmiGetRdmaInfoSize(int numOps){
  return LrtsGetRdmaInfoSize(numOps);
}

/* Set the ack function handler and token*/
void *CmiSetRdmaAck(RdmaAckHandlerFn fn, void *token){
  CmiRdmaAck *ack = (CmiRdmaAck *)malloc(sizeof(CmiRdmaAck));
  ack->fnPtr = fn;
  ack->token = token;
  return ack;
}


/* Receiver side functions */

/* Set the receiver specific machine layer info generic to RDMA ops*/
void CmiSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize){
  LrtsSetRdmaRecvInfo(dest, numOps, charmMsg, rdmaInfo, msgSize);
}

/* Set the receiver specific machine layer info specific to RDMA ops*/
void CmiSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaInfo){
  LrtsSetRdmaRecvOpInfo(dest, buffer, src_ref, size, opIndex, rdmaInfo);
}

/* Get the size occupied by the receiver specific machine layer specific to RDMA op*/
int CmiGetRdmaOpRecvInfoSize(void){
  return LrtsGetRdmaOpRecvInfoSize();
}

/* Get the size occupied by the receiver specific machine layer info generic to RDMA ops*/
int CmiGetRdmaGenRecvInfoSize(void){
  return LrtsGetRdmaGenRecvInfoSize();
}

/* Get the total size occupied by the receiver specific machine layer info*/
int CmiGetRdmaRecvInfoSize(int numOps){
  return LrtsGetRdmaRecvInfoSize(numOps);
}

/* Issue RDMA get calls on the pe using the message containing the metadata information*/
void CmiIssueRgets(void *recv, int pe){
  LrtsIssueRgets(recv, pe);
}

#endif

#if CMK_ONESIDED_DIRECT_IMPL

// Function Pointer to the individual Acknowledement handler function for the Direct API
RdmaSingleAckHandlerFn ncpyAckHandlerFn;

typedef struct _cmi_rdma_direct_ack {
  const void *srcAddr;
  int srcPe;
  const void *destAddr;
  int destPe;
  int ackSize;
} CmiRdmaDirectAck;

/* Support for Nocopy Direct API */
void LrtsSetRdmaSrcInfo(void *info, const void *ptr, int size);
void LrtsSetRdmaDestInfo(void *info, const void *ptr, int size);
void LrtsSetRdmaNcpyAck(RdmaAckHandlerFn fn);
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  int size);
void LrtsIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size);
void LrtsReleaseSourceResource(void *info, int pe);
void LrtsReleaseDestinationResource(void *info, int pe);

/* Set the machine specific information for a nocopy source pointer */
void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size){
  LrtsSetRdmaSrcInfo(info, ptr, size);
}

/* Set the machine specific information for a nocopy destination pointer */
void CmiSetRdmaDestInfo(void *info, const void *ptr, int size){
  LrtsSetRdmaDestInfo(info, ptr, size);
}

void *CmiGetNcpyAck(const void *srcAddr, void *srcAck, int srcPe, const void *destAddr, void *destAck, int destPe, int ackSize) {
  CmiRdmaDirectAck *directAck = (CmiRdmaDirectAck *)malloc(sizeof(CmiRdmaDirectAck) + 2*ackSize);
  directAck->srcAddr = srcAddr;
  directAck->srcPe = srcPe;
  directAck->destAddr = destAddr;
  directAck->destPe = destPe;
  directAck->ackSize = ackSize;

  // copy source ack
  memcpy((char *)directAck + sizeof(CmiRdmaDirectAck), srcAck, ackSize);

  // copy destination ack
  memcpy((char *)directAck + sizeof(CmiRdmaDirectAck) + ackSize, destAck, ackSize);

  return (void *)directAck;
}

void CmiInvokeNcpyAck(void *ack) {
  CmiRdmaDirectAck *directAck = (CmiRdmaDirectAck *)ack;

  // Retrieve source ack
  void *srcAck = (char *)directAck + sizeof(CmiRdmaDirectAck);

  // Retrieve destination ack
  void *destAck = (char *)srcAck + directAck->ackSize;

  ncpyAckHandlerFn(srcAck, directAck->srcPe, directAck->srcAddr);
  ncpyAckHandlerFn(destAck, directAck->destPe, directAck->destAddr);

  // free the allocated ack information
  free(directAck);
}

/* Set the ack handler function used in the Direct API */
void CmiSetRdmaNcpyAck(RdmaSingleAckHandlerFn fn){
  ncpyAckHandlerFn = fn;
}

/* Perform an RDMA Get operation into the local destination address from the remote source address*/
void CmiIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  int size) {

#if CMK_USE_CMA
  // check if remote PE is on the same physical node
  if(cma_works && CmiPeOnSamePhysicalNode(srcPe, destPe)) {

    CmiIssueRgetUsingCMA(srcAddr, srcInfo, srcPe,
                         destAddr, destInfo, destPe,
                         size);
    // directy invoke the acks
    ncpyAckHandlerFn(srcAck, srcPe, srcAddr);
    ncpyAckHandlerFn(destAck, destPe, destAddr);
    return;
  }
#endif
  // Use network RDMA for a PE on a remote host
  LrtsIssueRget(srcAddr,
                srcInfo,
                srcAck,
                srcAckSize,
                srcPe,
                destAddr,
                destInfo,
                destAck,
                destAckSize,
                destPe,
                size);
}

/* Perform an RDMA Put operation into the remote destination address from the local source address */
void CmiIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size) {

#if CMK_USE_CMA
  // check if remote PE is on the same physical node
  if(cma_works && CmiPeOnSamePhysicalNode(srcPe, destPe)) {

    CmiIssueRputUsingCMA(destAddr, destInfo, destPe,
                         srcAddr, srcInfo, srcPe,
                         size);

    // directy invoke the acks
    ncpyAckHandlerFn(srcAck, srcPe, srcAddr);
    ncpyAckHandlerFn(destAck, destPe, destAddr);
    return;
  }
#endif
  // Use network RDMA for a PE on a remote host
  LrtsIssueRput(destAddr,
                destInfo,
                destAck,
                destAckSize,
                destPe,
                srcAddr,
                srcInfo,
                srcAck,
                srcAckSize,
                srcPe,
                size);
}

/* Resource cleanup for source pointer */
void CmiReleaseSourceResource(void *info, int pe){
  LrtsReleaseSourceResource(info, pe);
}

/* Resource cleanup for destination pointer */
void CmiReleaseDestinationResource(void *info, int pe){
  LrtsReleaseDestinationResource(info, pe);
}

#endif /*End of CMK_ONESIDED_DIRECT_IMPL */
#endif
