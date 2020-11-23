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

static int rdma_device_send_handler_idx;

static void CmiRdmaDeviceSendHandler(DeviceRdmaOpMsg* msg) {
  DeviceRdmaOp* op = &msg->op;

  // Send device buffer through UCX with a special tag, so that it gets properly handled on the receiver
  CmiSendDevice(op);
}

void CmiRdmaDeviceSendInit() {
  // Register handler that initiates data transfer (sender -> receiver)
  rdma_device_send_handler_idx = CmiRegisterHandler((CmiHandler)CmiRdmaDeviceSendHandler);
}

void CmiRdmaDeviceIssueRget(DeviceRdmaOpMsg* msg, DeviceRdmaOp* op) {
  // Post a receive for device data
  CmiRecvDevice(op);

  // Send message with destination address to sender
  CmiSetHandler(msg, rdma_device_send_handler_idx);
  CmiSyncSendAndFree(msg->op.src_pe, sizeof(DeviceRdmaOpMsg), msg);
}

#include "machine-rdma.h"

void CmiSendDevice(DeviceRdmaOp* op) {
  LrtsSendDevice(op);
}

void CmiRecvDevice(DeviceRdmaOp* op) {
  LrtsRecvDevice(op);
}

RdmaAckHandlerFn rdmaDeviceRecvHandlerFn;

void CmiRdmaDeviceRecvInit(RdmaAckHandlerFn fn) {
  // Set handler function that gets invoked when data transfer is complete (on receiver)
  rdmaDeviceRecvHandlerFn = fn;
}

void CmiInvokeRecvHandler(void* data) {
  rdmaDeviceRecvHandlerFn(data);
}

#endif // CMK_CUDA
