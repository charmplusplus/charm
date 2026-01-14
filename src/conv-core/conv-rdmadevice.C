#include "converse.h"
#include "conv-rdmadevice.h"
#include "ck.h"

#if CMK_CUDA || CMK_HIP

CmiNcpyModeDevice findTransferModeDevice(int srcPe, int dstPe) {
  CmiEnforce((srcPe >= 0) && (srcPe <= CmiNumPes()));
  CmiEnforce((dstPe >= 0) && (dstPe <= CmiNumPes()));

  if (CmiNodeOf(srcPe) == CmiNodeOf(dstPe)) {
    // Same logical node
    return CmiNcpyModeDevice::MEMCPY;
  } else if (CmiPeOnSamePhysicalNode(srcPe, dstPe)) {
    // Different logical nodes, same physical node
    return CmiNcpyModeDevice::IPC;
  } else {
    // Different physical nodes, requires GPUDirect RDMA
    return CmiNcpyModeDevice::RDMA;
  }
}

#if CMK_GPU_COMM
#include "machine-rdma.h"

void CmiSendDevice(int& src_pe, const void*& ptr, size_t size, uint64_t& tag) {
  LrtsSendDevice(src_pe, ptr, size, tag);
}

void CmiRecvDevice(DeviceRdmaOp* op, DeviceRecvType type) {
  LrtsRecvDevice(op, type);
}

RdmaAckHandlerFn rdmaDeviceRecvHandlerFn;

void CmiRdmaDeviceRecvInit(RdmaAckHandlerFn fn) {
  // Set handler function that gets invoked when data transfer is complete (on receiver)
  rdmaDeviceRecvHandlerFn = fn;
}

void CmiInvokeRecvHandler(void* data) {
  QdProcess(1);
  rdmaDeviceRecvHandlerFn(data);
}
#endif // CMK_GPU_COMM
#endif // CMK_CUDA
