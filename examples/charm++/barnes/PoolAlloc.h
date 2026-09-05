#ifndef BARNES_POOL_ALLOC_H
#define BARNES_POOL_ALLOC_H

// Device allocation following the runtime's choice. Under +gpupool every
// device buffer here comes from CkDeviceMalloc -- a host-side buddy allocation
// out of a pre-allocated arena the peers have already opened, so neither the
// malloc nor the free touches the driver or synchronizes the device, and a
// device send from it never pays an IPC handle open. Without +gpupool it is
// the hapiMalloc it always was.
//
// One thing the swap changes: cudaFree synchronized the whole device, and a
// pool free does not. Every site that frees a buffer something may still be
// reading drains its stream first (GpuTraversalBatch::growDevice,
// TreePiece::launchDeviceWalk), which is what made those frees correct all
// along rather than a side effect of the driver.

#include "charm++.h"
#include "hapi.h"

inline hapiError_t bhMalloc(void** p, size_t n) {
  static const bool pool = CkDevicePoolOn();
  if (!pool) return hapiMalloc(p, n);
  *p = CkDeviceMalloc(n);
  return (*p != NULL) ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline hapiError_t bhFree(void* p) {
  static const bool pool = CkDevicePoolOn();
  if (p == NULL) return cudaSuccess;
  if (pool) { CkDeviceFree(p); return cudaSuccess; }
  return hapiFree(p);
}

#endif
