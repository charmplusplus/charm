#pragma once

// Named guard in addition to pragma once: this header is reachable through
// two paths (the source tree and the installed include directory), which
// pragma once treats as distinct files. The duplicate macros tolerated that;
// the hapiMallocRecord/hapiFreeRecord function definitions do not.
#ifndef HAPI_PORTABLE_H_SEEN
#define HAPI_PORTABLE_H_SEEN

#undef CMK_CUDA
#undef CMK_HIP

#include "conv-mach-opt.h"

#ifdef CMK_CUDA

#include <cuda_runtime.h>

#define hapiStream_t cudaStream_t

#define hapiEvent_t cudaEvent_t

#define hapiSetDevice(dev) cudaSetDevice(dev)

#define hapiPeekAtLastError cudaPeekAtLastError
#define hapiEventDefault cudaEventDefault
#define hapiEventDisableTiming cudaEventDisableTiming

#define hapiGetDeviceCount(devCount) cudaGetDeviceCount(devCount)

#define hapiDeviceCanAccessPeer(canAccess, dev1, dev2) \
    cudaDeviceCanAccessPeer(canAccess, dev1, dev2)

#define hapiDeviceEnablePeerAccess(dev, flags) \
    cudaDeviceEnablePeerAccess(dev, flags)

#define hapiEventCreateWithFlags(flags, event) cudaEventCreateWithFlags(flags, event)

// Records and stream waits go through hapi so that +gpustalltrace can log
// them: a stalled stream is then walked back to the wait that never came,
// and the record it was waiting for. Off, these are the plain CUDA calls
// behind one predictable branch. The noted forms carry a caller's tag (the
// IPC event slot, say) so a wait in one process can be matched to the
// record in another.
#ifdef __cplusplus
extern "C++" {
cudaError_t hapiEventRecordNoted(cudaEvent_t event, cudaStream_t stream, int note);
cudaError_t hapiStreamWaitEventNoted(cudaStream_t stream, cudaEvent_t event,
                                     unsigned int flags, int note);
}
#define hapiEventRecord(event, stream) hapiEventRecordNoted(event, stream, 0)
#define hapiStreamWaitEvent(stream, event, flags) \
    hapiStreamWaitEventNoted(stream, event, flags, 0)
#else
#define hapiEventRecord(event, stream) cudaEventRecord(event, stream)
#define hapiStreamWaitEvent(stream, event, flags) \
    cudaStreamWaitEvent(stream, event, flags)
#endif
#define hapiEventQuery(event) cudaEventQuery(event)
#define hapiEventDestroy(event) cudaEventDestroy(event)

#define hapiStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define hapiStreamCreate(stream) cudaStreamCreate(stream)
#define hapiStreamCreateNonBlocking(stream) \
    cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#define hapiStreamDestroy cudaStreamDestroy
#define hapiStreamDefault cudaStreamDefault
#define hapiStreamCreateWithPriority cudaStreamCreateWithPriority

#define hapiLaunchHostFunc(stream, func, args) \
    cudaLaunchHostFunc(stream, func, args)

#define hapiStreamPerThread cudaStreamPerThread

#define hapiIpcMemHandle_t cudaIpcMemHandle_t

#define hapiIpcEventHandle_t cudaIpcEventHandle_t

#define hapiIpcGetMemHandle(handle, ptr) cudaIpcGetMemHandle(handle, ptr)
#define hapiIpcCloseMemHandle(handle) cudaIpcCloseMemHandle(handle)

#define hapiIpcGetEventHandle(handle, event) cudaIpcGetEventHandle(handle, event)

#define hapiIpcOpenMemHandle(ptr, handle, flags) \
    cudaIpcOpenMemHandle(ptr, handle, flags)

#define hapiIpcOpenEventHandle(event, handle) \
    cudaIpcOpenEventHandle(event, handle)

#define hapiDeviceProp cudaDeviceProp

#define hapiGetDeviceProperties(prop, dev) cudaGetDeviceProperties(prop, dev)
#define hapiGetDevice(dev) cudaGetDevice(dev)

// Per-chare device footprint tracking (implemented in hapi_impl.cpp; no-ops
// when the load balancer is compiled out). hapiMalloc/hapiFree route through
// these so every allocation made inside an entry method is attributed to the
// running chare -- the producer behind the LB memory contract. Allocations
// outside entry methods (runtime startup, comm buffers) are left unattributed.
void hapiRecordAlloc(void* ptr, size_t size);
void hapiRecordFree(void* ptr);
// Drops a pointer's attribution without the IPC export invalidation
// hapiRecordFree does (a pool block, whose base is never freed).
void hapiRecordForget(void* ptr);

// Under +gpupool these are the device pool: hapiMalloc hands out a pool block
// (an arena the peers have opened and the fabric has registered, no driver
// call) and hapiFree returns one. Without the pool, cudaMalloc/cudaFree as
// ever. A pointer that predates the pool is still cudaFree'd. One difference
// to know: cudaFree waits for the device, a pool free does not -- use
// hapiFreeAsync(ptr, stream) to order the free behind a stream's work.
// Implemented in hapi_impl.cpp.
cudaError_t hapiMallocImpl(void** ptr, size_t size);
cudaError_t hapiFreeImpl(void* ptr);
cudaError_t hapiFreeAsync(void* ptr, cudaStream_t stream);

template <typename hapiMallocT>
static inline cudaError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  void* p = nullptr;
  const cudaError_t hapi_malloc_err = hapiMallocImpl(&p, size);
  if (hapi_malloc_err == cudaSuccess) *ptr = (hapiMallocT*)p;
  return hapi_malloc_err;
}
static inline cudaError_t hapiFreeRecord(void* ptr) { return hapiFreeImpl(ptr); }

#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeRecord(ptr)
// An allocation that belongs to the runtime whatever is running: an arena the
// pool or a comm buffer carves up, whose blocks are charged as they are handed
// out. Charging the arena too would bill a chare for all of it.
#define hapiMallocUnattributed(ptr, size) cudaMalloc((void**)(ptr), size)
#define hapiMallocHost(ptr, size) cudaMallocHost(ptr, size)
#define hapiFreeHost(ptr) cudaFreeHost(ptr)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    cudaHostGetDevicePointer(devPtr, hostPtr, flags)

#define hapiErrorMemoryAllocation cudaErrorMemoryAllocation
#define hapiErrorInitializationError cudaErrorInitializationError
#define hapiErrorAlreadyMapped cudaErrorAlreadyMapped
#define hapiSuccess cudaSuccess
#define hapiError_t cudaError_t

#define hapiMemcpyKind cudaMemcpyKind
#define hapiMemcpyHostToHost cudaMemcpyHostToHost
#define hapiMemcpyHostToDevice cudaMemcpyHostToDevice
#define hapiMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hapiMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hapiMemcpy(dst, src, count, kind) cudaMemcpy(dst, src, count, kind)

#define hapiGetErrorString(err) cudaGetErrorString(err)

#define hapiEventDisableTiming cudaEventDisableTiming
#define hapiEventInterprocess cudaEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync cudaMemcpyAsync
#define hapiMemcpy2DAsync cudaMemcpy2DAsync

#endif // CMK_CUDA

#ifdef CMK_HIP

#include <hip/hip_runtime.h>

#define hapiStream_t hipStream_t

#define hapiEvent_t hipEvent_t

#define hapiSetDevice(dev) hipSetDevice(dev)
#define hapiGetDeviceCount(devCount) hipGetDeviceCount(devCount)

#define hapiPeekAtLastError hipPeekAtLastError

#define hapiDeviceCanAccessPeer(canAccess, dev1, dev2) \
    hipDeviceCanAccessPeer(canAccess, dev1, dev2)
#define hapiDeviceEnablePeerAccess(dev, flags) \
    hipDeviceEnablePeerAccess(dev, flags)

#define hapiEventCreateWithFlags(flags, event) hipEventCreateWithFlags(flags, event)
#define hapiEventRecord(event, stream) hipEventRecord(event, stream)
#define hapiEventQuery(event) hipEventQuery(event)
#define hapiEventDestroy(event) hipEventDestroy(event)
#define hapiStreamWaitEvent(stream, event, flags) \
    hipStreamWaitEvent(stream, event, flags)

#define hapiStreamSynchronize(stream) hipStreamSynchronize(stream)

#define hapiLaunchHostFunc(stream, func, args) \
    hipLaunchHostFunc(stream, func, args)

#define hapiStreamPerThread hipStreamPerThread

#define hapiIpcMemHandle_t hipIpcMemHandle_t

#define hapiIpcEventHandle_t hipIpcEventHandle_t

#define hapiIpcGetMemHandle(handle, ptr) hipIpcGetMemHandle(handle, ptr)
#define hapiIpcCloseMemHandle(handle) hipIpcCloseMemHandle(handle)

#define hapiIpcGetEventHandle(handle, event) hipIpcGetEventHandle(handle, event)

#define hapiIpcOpenMemHandle(ptr, handle, flags) \
    hipIpcOpenMemHandle(ptr, handle, flags)

#define hapiIpcOpenEventHandle(event, handle) \
    hipIpcOpenEventHandle(event, handle)

#define hapiDeviceProp hipDeviceProp_t

#define hapiGetDeviceProperties(prop, dev) hipGetDeviceProperties(prop, dev)
#define hapiGetDevice(dev) hipGetDevice(dev)
#define hapiStreamCreate(stream) hipStreamCreate(stream)
#define hapiStreamCreateNonBlocking(stream) \
    hipStreamCreateWithFlags(stream, hipStreamNonBlocking)

// See the CUDA branch: allocation attribution for the LB memory contract.
void hapiRecordAlloc(void* ptr, size_t size);
void hapiRecordFree(void* ptr);
void hapiRecordForget(void* ptr);

// See the CUDA branch: the device pool when +gpupool is on.
hipError_t hapiMallocImpl(void** ptr, size_t size);
hipError_t hapiFreeImpl(void* ptr);
hipError_t hapiFreeAsync(void* ptr, hipStream_t stream);

template <typename hapiMallocT>
static inline hipError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  void* p = nullptr;
  const hipError_t hapi_malloc_err = hapiMallocImpl(&p, size);
  if (hapi_malloc_err == hipSuccess) *ptr = (hapiMallocT*)p;
  return hapi_malloc_err;
}
static inline hipError_t hapiFreeRecord(void* ptr) { return hapiFreeImpl(ptr); }

#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeRecord(ptr)
#define hapiMallocUnattributed(ptr, size) hipMalloc((void**)(ptr), size)
#define hapiMallocHost(ptr, size) hipHostMalloc(ptr, size)
#define hapiFreeHost(ptr) hipFreeHost(ptr)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    hipHostGetDevicePointer(devPtr, hostPtr, flags)

#define hapiErrorMemoryAllocation hipErrorMemoryAllocation
#define hapiErrorInitializationError hipErrorInitializationError
#define hapiErrorAlreadyMapped hipErrorAlreadyMapped
#define hapiSuccess hipSuccess
#define hapiError_t hipError_t
#define hapiStreamDestroy hipStreamDestroy
#define hapiStreamDefault hipStreamDefault
#define hapiStreamCreateWithPriority hipStreamCreateWithPriority

#define hapiMemcpyKind hipMemcpyKind
#define hapiMemcpyHostToHost hipMemcpyHostToHost
#define hapiMemcpyHostToDevice hipMemcpyHostToDevice
#define hapiMemcpyDeviceToHost hipMemcpyDeviceToHost
#define hapiMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define hapiMemcpy(dst, src, count, kind) hipMemcpy(dst, src, count, kind)
#define hapiGetErrorString(err) hipGetErrorString(err)

#define hapiEventDisableTiming hipEventDisableTiming
#define hapiEventInterprocess hipEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess


#endif // CMK_HIP

#endif // HAPI_PORTABLE_H_SEEN
