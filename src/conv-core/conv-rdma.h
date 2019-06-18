#ifndef _CONV_RDMA_H
#define _CONV_RDMA_H

#include "cmirdmautils.h"

typedef void (*RdmaAckCallerFn)(void *token);

/* Support for Direct API */
void CmiSetRdmaCommonInfo(void *info, const void *ptr, int size);
int CmiGetRdmaCommonInfoSize(void);

void CmiSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);

// Function to set the ack handler for the Direct API
void CmiSetDirectNcpyAckHandler(RdmaAckCallerFn fn);

/* CmiIssueRget initiates an RDMA read operation, transferring 'size' bytes of data from the address space of 'srcPe' to local address, 'destAddr'.
 * When the runtime invokes srcAck on the source (target), it indicates safety to overwrite or free the srcAddr buffer.
 * When the runtime invokes destAck on the destination (initiator), it indicates that the data has been successfully received in the
 * destAddr buffer.
 */
void CmiIssueRget(NcpyOperationInfo *ncpyOpInfo);

/* CmiIssueRput initiates an RDMA write operation, transferring 'size' bytes of data from the local address, 'srcAddr' to the address space of 'destPe'.
 * When the runtime invokes srcAck on the source (initiator), it indicates safety to overwrite or free the srcAddr buffer.
 * When the runtime invokes destAck on the destination (target), it indicates that the data has been successfully received in the
 * destAddr buffer.
 */

void CmiIssueRput(NcpyOperationInfo *ncpyOpInfo);

void CmiDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode);

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

// Allocation from pool
void *CmiRdmaAlloc(int size);

int CmiDoesCMAWork(void);

#if !CMK_ONESIDED_IMPL
// Function declaration for supporting generic Direct Nocopy API
void CmiOnesidedDirectInit(void);
#endif

#endif
