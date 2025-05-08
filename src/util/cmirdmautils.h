#ifndef _CKRDMAUTILS_H
#define _CKRDMAUTILS_H

//#include "conv-header.h"
#include "converse.h"
#include <stdio.h>
#include <stddef.h>

#if CMK_CUDA
enum DeviceRecvType {
  DEVICE_RECV_TYPE_CHARM,
  DEVICE_RECV_TYPE_AMPI,
  DEVICE_RECV_TYPE_CHARM4PY
};

typedef struct DeviceRdmaInfo_ {
  int n_ops; // Number of RDMA operations, i.e. number of buffers being sent
  int counter; // Used to track the number of completed RDMA operations
  void* msg; // Charm++ message to be (re-)enqueued after all operations complete
} DeviceRdmaInfo;

typedef struct DeviceRdmaOp_ {
  int dest_pe;
  const void* dest_ptr;
  size_t size;
  DeviceRdmaInfo* info;
  void* src_cb;
  void* dst_cb;
  uint64_t tag;
} DeviceRdmaOp;

typedef struct DeviceRdmaOpMsg_ {
  char header[CmiMsgHeaderSizeBytes];
  DeviceRdmaOp op;
} DeviceRdmaOpMsg;
#endif // CMK_CUDA

#ifdef __cplusplus
extern "C" {
#endif

int getNcpyOpInfoTotalSize(
  int srcLayerSize,
  int srcAckSize,
  int destLayerSize,
  int destAckSize);

void setNcpyOpInfo(
  const void *srcPtr,
  char *srcLayerInfo,
  int srcLayerSize,
  char *srcAck,
  int srcAckSize,
  size_t srcSize,
  unsigned short int srcRegMode,
  unsigned short int srcDeregMode,
  unsigned short int isSrcRegistered,
  int srcPe,
  const void *srcRef,
  const void *destPtr,
  char *destLayerInfo,
  int destLayerSize,
  char *destAck,
  int destAckSize,
  size_t destSize,
  unsigned short int destRegMode,
  unsigned short int destDeregMode,
  unsigned short int isdestRegistered,
  int destPe,
  const void *destRef,
  int rootNode,
  NcpyOperationInfo *ncpyOpInfo);

void resetNcpyOpInfoPointers(NcpyOperationInfo *ncpyOpInfo);

void setReverseModeForNcpyOpInfo(NcpyOperationInfo *ncpyOpInfo);

#ifdef __cplusplus
}
#endif

#endif
