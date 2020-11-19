#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

/* Support for Nocopy Direct API */
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode);
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo);

void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo);

void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode);

#if CMK_REG_REQUIRED
void LrtsInvokeRemoteDeregAckHandler(int pe, NcpyOperationInfo *ncpyOpInfo);
#endif

void CmiInvokeNcpyAck(void *ack);

#if CMK_CUDA
void LrtsSendDevice(DeviceRdmaOp* op);
void LrtsRecvDevice(DeviceRdmaOp* op);

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
#endif

int CmiGetRdmaCommonInfoSize();
#endif
