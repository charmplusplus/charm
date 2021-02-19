#include "converse.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA

CmiNcpyModeDevice findTransferModeDevice(int srcPe, int destPe) {
  if (CmiNodeOf(srcPe) == CmiNodeOf(destPe)) {
    // Same logical node
    return CmiNcpyModeDevice::MEMCPY;
  }
  else if (CmiPeOnSamePhysicalNode(srcPe, destPe)) {
    // Different logical nodes, same physical node
    return CmiNcpyModeDevice::IPC;
  }
  else {
    // Different physical nodes, requires GPUDirect RDMA
    return CmiNcpyModeDevice::RDMA;
  }
}

#include "machine-rdma.h"

void CmiSendDevice(int dest_pe, const void*& ptr, size_t size, uint64_t& tag) {
  LrtsSendDevice(dest_pe, ptr, size, tag);
}

void CmiRecvDevice(DeviceRdmaOp* op, DeviceRecvType type) {
  LrtsRecvDevice(op, type);
}

RdmaAckHandlerFn rdmaDeviceRecvHandlerFn;
RdmaAckHandlerFn rdmaDeviceAmpiRecvHandlerFn;
#if CMK_CHARM4PY
RdmaAckHandlerFn rdmaDeviceExtRecvHandlerFn;

void CmiRdmaDeviceRecvInit(RdmaAckHandlerFn fn1, RdmaAckHandlerFn fn2, RdmaAckHandlerFn fn3) {
  // Set handler function that gets invoked when data transfer is complete (on receiver)
  rdmaDeviceRecvHandlerFn = fn1;
  rdmaDeviceAmpiRecvHandlerFn = fn2;
  rdmaDeviceExtRecvHandlerFn = fn3;
}
#else
void CmiRdmaDeviceRecvInit(RdmaAckHandlerFn fn1, RdmaAckHandlerFn fn2) {
  // Set handler function that gets invoked when data transfer is complete (on receiver)
  rdmaDeviceRecvHandlerFn = fn1;
  rdmaDeviceAmpiRecvHandlerFn = fn2;
}
#endif // CMK_CHARM4PY

void CmiInvokeRecvHandler(void* data) {
  #if CMK_CHARMPY
  rdmaDeviceExtRecvHandlerFn(data);
  #else
  rdmaDeviceRecvHandlerFn(data);
  #endif
}

void CmiInvokeAmpiRecvHandler(void* data) {
  rdmaDeviceAmpiRecvHandlerFn(data);
}

#if CMK_CHARM4PY
void CmiInvokeExtRecvHandler(void* data) {
  rdmaDeviceExtRecvHandlerFn(data);
}
#endif
#endif // CMK_CUDA
