#ifndef _CKRDMAUTILS_H
#define _CKRDMAUTILS_H

#include "charm-config.h"
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
  // Timed tally (CHARM_ZC_STATS), zero when it is off. Carried here rather than
  // in the stall watch because this struct already lives exactly from the
  // moment the receive is posted to the moment its last op completes, which is
  // the interval the measurement wants; the watch is gated on a different env
  // var and would tie the two together.
  double zc_posted;   // CkWallTimer() when the receive was issued
  size_t zc_bytes;    // total bytes across all ops
  int zc_mode;        // CkNcpyModeDevice the ops resolved to
} DeviceRdmaInfo;

typedef struct DeviceRdmaOp_ {
  // The PE's receive stream, where this receive's copy lands. Every device
  // receive on a PE uses the one stream; a transfer deferred by the
  // migration-mismatch correction resumes on it too.
  void* stream;
  const void* dest_ptr;
  size_t size;
  DeviceRdmaInfo* info;
  void* src_cb;
  void* dst_cb;
  uint64_t tag;
  int dest_pe;
  // Receiving element. The stand-down its receives took is released when the
  // completion message dies (see the consumption-hold map in ckrdmadevice.C),
  // so completion handlers themselves never release it; these fields let a
  // deferred correction name the element it is resolving for.
  int dest_aid_idx;
  CmiUInt8 dest_id;
  int src_pe;
  int src_mpi_rank;
  int dest_mpi_rank;
  // Per-op timing for the cross-node tier (CHARM_ZC_STATS), zero when off.
  // The per-message tally in DeviceRdmaInfo only records once every op of a
  // receive has completed; an rget issued alongside an op that resolves some
  // other way never reaches it, which is why the RDMA slot printed no row at
  // all while its mode counter showed 60480 receives per process. This one
  // is stamped where the rget is issued and read where it completes, so it
  // cannot be lost to the aggregation.
  double rget_posted;
  // What the destination still had to wait for when the receive was posted:
  // the work issued on the posted stream before the post. dst_flag_seq names
  // the pinned flag issued on that stream then (0: none), dst_event an event
  // recorded there when the flag ring had no slot (NULL: none). Both zero
  // means the destination was already free at post time. Decided at post so
  // a receive deferred for correction resumes with the same condition.
  int dst_flag_rank;
  uint32_t dst_flag_seq;
  void* dst_event;
} DeviceRdmaOp;

typedef struct DeviceRdmaOpMsg_ {
  char header[CmiMsgHeaderSizeBytes];
  DeviceRdmaOp* op;
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
