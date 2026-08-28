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

#define hapiEventRecord(event, stream) cudaEventRecord(event, stream)
#define hapiEventQuery(event) cudaEventQuery(event)
#define hapiEventDestroy(event) cudaEventDestroy(event)
#define hapiStreamWaitEvent(stream, event, flags) \
    cudaStreamWaitEvent(stream, event, flags)

#define hapiStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define hapiStreamCreate(stream) cudaStreamCreate(stream)
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

template <typename hapiMallocT>
static inline cudaError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  const cudaError_t hapi_malloc_err = cudaMalloc((void**)ptr, size);
  if (hapi_malloc_err == cudaSuccess) hapiRecordAlloc((void*)*ptr, size);
  return hapi_malloc_err;
}
static inline cudaError_t hapiFreeRecord(void* ptr) {
  hapiRecordFree(ptr);
  return cudaFree(ptr);
}

#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeRecord(ptr)
#define hapiMallocHost(ptr, size) cudaMallocHost(ptr, size)
#define hapiFreeHost(ptr) cudaFreeHost(ptr)

#define hapiErrorMemoryAllocation cudaErrorMemoryAllocation
#define hapiErrorInitializationError cudaErrorInitializationError
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

// See the CUDA branch: allocation attribution for the LB memory contract.
void hapiRecordAlloc(void* ptr, size_t size);
void hapiRecordFree(void* ptr);

template <typename hapiMallocT>
static inline hipError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  const hipError_t hapi_malloc_err = hipMalloc((void**)ptr, size);
  if (hapi_malloc_err == hipSuccess) hapiRecordAlloc((void*)*ptr, size);
  return hapi_malloc_err;
}
static inline hipError_t hapiFreeRecord(void* ptr) {
  hapiRecordFree(ptr);
  return hipFree(ptr);
}

#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeRecord(ptr)
#define hapiMallocHost(ptr, size) hipHostMalloc(ptr, size)
#define hapiFreeHost(ptr) hipFreeHost(ptr)

#define hapiErrorMemoryAllocation hipErrorMemoryAllocation
#define hapiErrorInitializationError hipErrorInitializationError
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
