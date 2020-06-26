#if CMK_CUDA

#include "converse.h"
#include "conv-rdmadevice.h"

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

#endif // CMK_CUDA
