#ifndef _CONV_RDMA_H
#define _CONV_RDMA_H

typedef void (*RdmaSingleAckCallerFn)(void *cbPtr, int pe, const void *ptr);
typedef void (*RdmaAckCallerFn)(void *token);

void *CmiSetRdmaAck(RdmaAckCallerFn fn, void *token);
void CmiSetRdmaInfo(void *dest, int destPE, int numOps);
void CmiSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE);
int CmiGetRdmaOpInfoSize(void);
int CmiGetRdmaGenInfoSize(void);

int CmiGetRdmaInfoSize(int numOps);
void CmiSetRdmaRecvInfo(void *dest, int numOps, void *msg, void *rdmaInfo, int msgSize);
void CmiSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaInfo);
int CmiGetRdmaOpRecvInfoSize(void);
int CmiGetRdmaGenRecvInfoSize(void);
int CmiGetRdmaRecvInfoSize(int numOps);

void CmiIssueRgets(void *recv, int pe);

/* Support for Direct API */
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size);
int CmiGetRdmaCommonInfoSize(void);

void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size);
void CmiSetRdmaDestInfo(void *info, const void *ptr, int size);
void CmiSetRdmaNcpyAck(RdmaSingleAckCallerFn fn);

/* CmiIssueRget initiates an RDMA read operation, transferring 'size' bytes of data from the address space of 'srcPe' to local address, 'destAddr'.
 * When the runtime invokes srcAck on the source (target), it indicates safety to overwrite or free the srcAddr buffer.
 * When the runtime invokes destAck on the destination (initiator), it indicates that the data has been successfully received in the
 * destAddr buffer.
 */
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
  int size);

/* CmiIssueRput initiates an RDMA write operation, transferring 'size' bytes of data from the local address, 'srcAddr' to the address space of 'destPe'.
 * When the runtime invokes srcAck on the source (initiator), it indicates safety to overwrite or free the srcAddr buffer.
 * When the runtime invokes destAck on the destination (target), it indicates that the data has been successfully received in the
 * destAddr buffer.
 */

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
  int size);

void CmiReleaseSourceResource(void *info, int pe);
void CmiReleaseDestinationResource(void *info, int pe);

#if CMK_USE_CMA
void CmiIssueRgetUsingCMA(
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  const void* destAddr,
  void *destInfo,
  int destPe,
  int size);

void CmiIssueRputUsingCMA(
  const void* destAddr,
  void *destInfo,
  int destPe,
  const void* srcAddr,
  void *srcInfo,
  int srcPe,
  int size);
#endif

#if !CMK_ONESIDED_DIRECT_IMPL
// Function declaration used for the generic implementation of the Nocopy Direct API
void CmiOnesidedDirectInit(void);
#endif
#endif
