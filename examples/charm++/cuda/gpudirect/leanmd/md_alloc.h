#ifndef __MD_ALLOC_H__
#define __MD_ALLOC_H__

// Device allocation for leanmd's chares, following the runtime's choice.
//
// Under +gpupool every buffer here comes from CkDeviceMalloc: a host-side
// buddy allocation out of a pre-allocated arena, so neither the malloc nor
// the free touches the driver or synchronizes the device, and every buffer
// sits inside an arena the peers have already opened, so a device send from
// it never pays an IPC handle open. Without +gpupool everything comes from
// hapiMalloc and nothing else changes. One binary, one switch.

#include "charm++.h"
#include "hapi.h"

inline bool mdPoolOn() {
  static const bool on = CkDevicePoolOn();
  return on;
}

// A buffer that never migrates (a Cell's): pool or hapiMalloc, freed the same
// way it was made.
inline hapiError_t mdMalloc(void** p, size_t n) {
  if (!mdPoolOn()) return hapiMalloc(p, n);
  *p = CkDeviceMalloc(n);
  return (*p != NULL) ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline hapiError_t mdFree(void* p) {
  if (p == NULL) return cudaSuccess;
  if (mdPoolOn()) { CkDeviceFree(p); return cudaSuccess; }
  return hapiFree(p);
}

// A buffer that travels with its chare through pup_buffer_device (a
// Compute's). After a migration the pointer aims into the runtime's landing
// arena rather than at an allocation of ours, and hapiFreeMigratable is the
// one call that is correct for both a hapiMalloc'd buffer and a rebound one.
// fromPool is set only when this chare took the buffer from the pool itself.
inline hapiError_t mdMigMalloc(void** p, size_t n) { return mdMalloc(p, n); }
inline void mdMigFree(void* p, bool fromPool) {
  if (p == NULL) return;
  if (fromPool) CkDeviceFree(p); else hapiFreeMigratable(p);
}

#endif
