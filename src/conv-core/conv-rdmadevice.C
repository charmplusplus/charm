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

void CmiRecvDevice(DeviceRdmaOp* op, bool ampi) {
  LrtsRecvDevice(op, ampi);
}

RdmaAckHandlerFn rdmaDeviceRecvHandlerFn;
RdmaAckHandlerFn rdmaDeviceAmpiRecvHandlerFn;

void CmiRdmaDeviceRecvInit(RdmaAckHandlerFn fn1, RdmaAckHandlerFn fn2) {
  // Set handler function that gets invoked when data transfer is complete (on receiver)
  rdmaDeviceRecvHandlerFn = fn1;
  rdmaDeviceAmpiRecvHandlerFn = fn2;
}

void CmiInvokeRecvHandler(void* data) {
  rdmaDeviceRecvHandlerFn(data);
}

void CmiInvokeAmpiRecvHandler(void* data) {
  rdmaDeviceAmpiRecvHandlerFn(data);
}

#endif // CMK_CUDA
