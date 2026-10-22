#pragma once

// Named guard in addition to pragma once: this header is reachable through
// two paths (the source tree and the installed include directory), which
// pragma once treats as distinct files. The duplicate macros tolerated that;
// the function definitions below do not.
#ifndef HAPI_PORTABLE_H_SEEN
#define HAPI_PORTABLE_H_SEEN

// The vendor table. Every GPU call the runtime makes goes through a hapi
// name defined here for both vendors; a source file that spells a cuda* or
// hip* name is vendor-specific and says so with CMK_CUDA / CMK_HIP. The two
// tables are kept in the same order so a name missing from one is easy to see.
//
// Three kinds of entry:
//  - runtime API names (streams, events, copies, IPC): the same call on both;
//  - driver-level names (virtual memory, completion-flag writers, address
//    ranges): CUDA's driver API against HIP's runtime, so these are inline
//    functions taking integer device addresses, which hides CUdeviceptr being
//    an integer and hipDeviceptr_t a pointer;
//  - constants and types.

#undef CMK_CUDA
#undef CMK_HIP

#include "conv-mach-opt.h"
#include <stdint.h>
#include <stddef.h>

#if CMK_CUDA

#include <cuda_runtime.h>
#include <cuda.h>   // driver API: the virtual-memory pool, address ranges, flag writers

// ---- types ----
#define hapiStream_t cudaStream_t
#define hapiEvent_t cudaEvent_t
#define hapiError_t cudaError_t
#define hapiMemcpyKind cudaMemcpyKind
#define hapiDeviceProp cudaDeviceProp
#define hapiDeviceAttr_t cudaDeviceAttr
#define hapiIpcMemHandle_t cudaIpcMemHandle_t
#define hapiIpcEventHandle_t cudaIpcEventHandle_t
#define hapiPointerAttributes cudaPointerAttributes
#define hapiHostFn_t cudaHostFn_t
#define hapiDrvResult CUresult
#define hapiDrvDevice CUdevice
#define hapiDrvStream CUstream
#define hapiDrvDeviceptr CUdeviceptr
#define hapiMemGenericAllocationHandle CUmemGenericAllocationHandle
#define hapiMemAllocationProp CUmemAllocationProp
#define hapiMemAccessDesc CUmemAccessDesc
#define hapiMemAllocationHandleType CUmemAllocationHandleType
#define hapiMemAllocationGranularityFlags CUmemAllocationGranularity_flags

// ---- constants ----
#define hapiSuccess cudaSuccess
#define hapiErrorNotReady cudaErrorNotReady
#define hapiErrorInvalidResourceHandle cudaErrorInvalidResourceHandle
#define hapiErrorMemoryAllocation cudaErrorMemoryAllocation
#define hapiErrorNotSupported cudaErrorNotSupported
#define hapiErrorAlreadyMapped cudaErrorAlreadyMapped
#define hapiErrorInitializationError cudaErrorInitializationError
#define hapiMemcpyDefault cudaMemcpyDefault
#define hapiMemcpyHostToHost cudaMemcpyHostToHost
#define hapiMemcpyHostToDevice cudaMemcpyHostToDevice
#define hapiMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hapiMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hapiStreamNonBlocking cudaStreamNonBlocking
#define hapiStreamDefault cudaStreamDefault
#define hapiStreamPerThread cudaStreamPerThread
#define hapiEventDefault cudaEventDefault
#define hapiEventDisableTiming cudaEventDisableTiming
#define hapiEventInterprocess cudaEventInterprocess
#define hapiHostAllocPortable cudaHostAllocPortable
#define hapiHostAllocMapped cudaHostAllocMapped
#define hapiIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define hapiMemoryTypeUnregistered cudaMemoryTypeUnregistered
#define hapiMemoryTypeHost cudaMemoryTypeHost
#define hapiMemoryTypeDevice cudaMemoryTypeDevice
#define hapiMemoryTypeManaged cudaMemoryTypeManaged
#define hapiDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define hapiDevAttrMaxBlocksPerMultiprocessor cudaDevAttrMaxBlocksPerMultiprocessor
#define hapiDevAttrClockRate cudaDevAttrClockRate
#define hapiDevAttrMaxThreadsPerMultiProcessor cudaDevAttrMaxThreadsPerMultiProcessor
#define hapiDevAttrMaxRegistersPerMultiprocessor cudaDevAttrMaxRegistersPerMultiprocessor
#define hapiDevAttrMaxSharedMemoryPerMultiprocessor cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define hapiDevAttrWarpSize cudaDevAttrWarpSize
#define hapiDrvSuccess CUDA_SUCCESS
#define hapiDrvErrorNotSupported CUDA_ERROR_NOT_SUPPORTED
#define hapiStreamWriteValueDefault CU_STREAM_WRITE_VALUE_DEFAULT
#define hapiMemHandleTypePosixFd CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#define hapiMemHandleTypeNone CU_MEM_HANDLE_TYPE_NONE
#define hapiMemLocationTypeDevice CU_MEM_LOCATION_TYPE_DEVICE
#define hapiMemAllocationTypePinned CU_MEM_ALLOCATION_TYPE_PINNED
#define hapiMemAccessFlagsProtReadWrite CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define hapiMemAllocGranularityMinimum CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define hapiMemAllocGranularityRecommended CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
// The environment variable the driver reads at init for the number of
// hardware work queues per device (hapiPresetEnvironment).
#define HAPI_HW_QUEUES_ENV "CUDA_DEVICE_MAX_CONNECTIONS"
#define HAPI_HW_QUEUES_DRIVER_DEFAULT "8"
#define HAPI_VENDOR_NAME "CUDA"

// ---- runtime API ----
#define hapiGetLastError cudaGetLastError
#define hapiPeekAtLastError cudaPeekAtLastError
#define hapiGetErrorName cudaGetErrorName
#define hapiGetErrorString(err) cudaGetErrorString(err)
#define hapiErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define hapiSetDevice(dev) cudaSetDevice(dev)
#define hapiGetDevice(dev) cudaGetDevice(dev)
#define hapiGetDeviceCount(devCount) cudaGetDeviceCount(devCount)
#define hapiGetDeviceProperties(prop, dev) cudaGetDeviceProperties(prop, dev)
#define hapiDeviceGetAttribute cudaDeviceGetAttribute
#define hapiDeviceSynchronize cudaDeviceSynchronize
#define hapiDeviceCanAccessPeer(canAccess, dev1, dev2) \
    cudaDeviceCanAccessPeer(canAccess, dev1, dev2)
#define hapiDeviceEnablePeerAccess(dev, flags) \
    cudaDeviceEnablePeerAccess(dev, flags)
#define hapiDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define hapiMemGetInfo cudaMemGetInfo
#define hapiMallocRaw(ptr, size) cudaMalloc((void**)(ptr), size)
#define hapiFreeRaw(ptr) cudaFree(ptr)
#define hapiMallocHost(ptr, size) cudaMallocHost(ptr, size)
#define hapiFreeHost(ptr) cudaFreeHost(ptr)
#define hapiHostAlloc(ptr, size, flags) cudaHostAlloc(ptr, size, flags)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    cudaHostGetDevicePointer(devPtr, hostPtr, flags)
#define hapiMemcpy(dst, src, count, kind) cudaMemcpy(dst, src, count, kind)
#define hapiMemcpyAsync cudaMemcpyAsync
#define hapiMemcpy2DAsync cudaMemcpy2DAsync
#define hapiMemsetAsync cudaMemsetAsync
#define hapiMemcpyToSymbol(sym, src, n) cudaMemcpyToSymbol(sym, src, n)
#define hapiStreamCreate(stream) cudaStreamCreate(stream)
#define hapiStreamCreateNonBlocking(stream) \
    cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#define hapiStreamCreateWithFlags cudaStreamCreateWithFlags
#define hapiStreamCreateWithPriority cudaStreamCreateWithPriority
#define hapiStreamDestroy cudaStreamDestroy
#define hapiStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define hapiStreamQuery cudaStreamQuery
#define hapiEventCreate cudaEventCreate
#define hapiEventCreateWithFlags(flags, event) cudaEventCreateWithFlags(flags, event)
#define hapiEventQuery(event) cudaEventQuery(event)
#define hapiEventSynchronize cudaEventSynchronize
#define hapiEventDestroy(event) cudaEventDestroy(event)
#define hapiEventElapsedTime(ms, start, end) cudaEventElapsedTime(ms, start, end)
#define hapiMemset(ptr, value, count) cudaMemset(ptr, value, count)
#define hapiLaunchHostFunc(stream, func, args) \
    cudaLaunchHostFunc(stream, func, args)
#define hapiLaunchKernelRaw cudaLaunchKernel
#define hapiIpcGetMemHandle(handle, ptr) cudaIpcGetMemHandle(handle, ptr)
#define hapiIpcCloseMemHandle(handle) cudaIpcCloseMemHandle(handle)
#define hapiIpcGetEventHandle(handle, event) cudaIpcGetEventHandle(handle, event)
#define hapiIpcOpenMemHandle(ptr, handle, flags) \
    cudaIpcOpenMemHandle(ptr, handle, flags)
#define hapiIpcOpenEventHandle(event, handle) \
    cudaIpcOpenEventHandle(event, handle)
#define hapiPointerGetAttributes cudaPointerGetAttributes
#define hapiMallocAsyncRaw cudaMallocAsync
#define hapiFreeAsyncRaw cudaFreeAsync
// The plain record and wait, for the implementation of the noted forms below.
#define hapiEventRecordRaw cudaEventRecord
#define hapiStreamWaitEventRaw cudaStreamWaitEvent

// ---- driver level: integer device addresses on both vendors ----
#define HAPI_DEVPTR(x) ((CUdeviceptr)(uintptr_t)(x))
static inline hapiDrvResult hapiDrvGetErrorName(hapiDrvResult r, const char** s) { return cuGetErrorName(r, s); }
static inline hapiDrvResult hapiDrvDeviceGet(hapiDrvDevice* d, int ordinal) { return cuDeviceGet(d, ordinal); }
static inline hapiDrvResult hapiDrvDeviceTotalMem(size_t* bytes, hapiDrvDevice d) { return cuDeviceTotalMem(bytes, d); }
// Whether the device can back a reserved range with physical chunks, export
// them as file descriptors, and have them registered by the NIC.
static inline void hapiDrvDeviceVmmSupport(hapiDrvDevice d, int* vmm, int* fd, int* rdma) {
  *vmm = *fd = *rdma = 0;
  cuDeviceGetAttribute(vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d);
  cuDeviceGetAttribute(fd, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, d);
  cuDeviceGetAttribute(rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, d);
}
static inline hapiDrvResult hapiDrvMemAddressReserve(uintptr_t* out, size_t size, size_t align,
                                                      uintptr_t hint, unsigned long long flags) {
  CUdeviceptr p = 0;
  const CUresult r = cuMemAddressReserve(&p, size, align, (CUdeviceptr)hint, flags);
  *out = (uintptr_t)p;
  return r;
}
static inline hapiDrvResult hapiDrvMemAddressFree(uintptr_t at, size_t size) { return cuMemAddressFree((CUdeviceptr)at, size); }
static inline hapiDrvResult hapiDrvMemCreate(hapiMemGenericAllocationHandle* h, size_t size,
                                             const hapiMemAllocationProp* prop, unsigned long long flags) {
  return cuMemCreate(h, size, prop, flags);
}
static inline hapiDrvResult hapiDrvMemRelease(hapiMemGenericAllocationHandle h) { return cuMemRelease(h); }
static inline hapiDrvResult hapiDrvMemMap(uintptr_t at, size_t size, size_t offset,
                                          hapiMemGenericAllocationHandle h, unsigned long long flags) {
  return cuMemMap((CUdeviceptr)at, size, offset, h, flags);
}
static inline hapiDrvResult hapiDrvMemUnmap(uintptr_t at, size_t size) { return cuMemUnmap((CUdeviceptr)at, size); }
static inline hapiDrvResult hapiDrvMemSetAccess(uintptr_t at, size_t size, const hapiMemAccessDesc* d, size_t n) {
  return cuMemSetAccess((CUdeviceptr)at, size, d, n);
}
static inline hapiDrvResult hapiDrvMemExportToShareableHandle(void* out, hapiMemGenericAllocationHandle h,
                                                              hapiMemAllocationHandleType t, unsigned long long flags) {
  return cuMemExportToShareableHandle(out, h, t, flags);
}
static inline hapiDrvResult hapiDrvMemImportFromShareableHandle(hapiMemGenericAllocationHandle* h, void* os,
                                                                hapiMemAllocationHandleType t) {
  return cuMemImportFromShareableHandle(h, os, t);
}
static inline hapiDrvResult hapiDrvMemGetAllocationGranularity(size_t* g, const hapiMemAllocationProp* p,
                                                               hapiMemAllocationGranularityFlags f) {
  return cuMemGetAllocationGranularity(g, p, f);
}

#elif CMK_HIP

// The HIP headers need the platform named; hipcc does it, a host compiler
// building an application through charmc does not.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>

// ---- types ----
#define hapiStream_t hipStream_t
#define hapiEvent_t hipEvent_t
#define hapiError_t hipError_t
#define hapiMemcpyKind hipMemcpyKind
#define hapiDeviceProp hipDeviceProp_t
#define hapiDeviceAttr_t hipDeviceAttribute_t
#define hapiIpcMemHandle_t hipIpcMemHandle_t
#define hapiIpcEventHandle_t hipIpcEventHandle_t
#define hapiPointerAttributes hipPointerAttribute_t
#define hapiHostFn_t hipHostFn_t
#define hapiDrvResult hipError_t
#define hapiDrvDevice hipDevice_t
#define hapiDrvStream hipStream_t
#define hapiDrvDeviceptr hipDeviceptr_t
#define hapiMemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define hapiMemAllocationProp hipMemAllocationProp
#define hapiMemAccessDesc hipMemAccessDesc
#define hapiMemAllocationHandleType hipMemAllocationHandleType
#define hapiMemAllocationGranularityFlags hipMemAllocationGranularity_flags

// ---- constants ----
#define hapiSuccess hipSuccess
#define hapiErrorNotReady hipErrorNotReady
#define hapiErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define hapiErrorMemoryAllocation hipErrorMemoryAllocation
#define hapiErrorNotSupported hipErrorNotSupported
#define hapiErrorAlreadyMapped hipErrorAlreadyMapped
#define hapiErrorInitializationError hipErrorInitializationError
#define hapiMemcpyDefault hipMemcpyDefault
#define hapiMemcpyHostToHost hipMemcpyHostToHost
#define hapiMemcpyHostToDevice hipMemcpyHostToDevice
#define hapiMemcpyDeviceToHost hipMemcpyDeviceToHost
#define hapiMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define hapiStreamNonBlocking hipStreamNonBlocking
#define hapiStreamDefault hipStreamDefault
#define hapiStreamPerThread hipStreamPerThread
#define hapiEventDefault hipEventDefault
#define hapiEventDisableTiming hipEventDisableTiming
#define hapiEventInterprocess hipEventInterprocess
#define hapiHostAllocPortable hipHostMallocPortable
#define hapiHostAllocMapped hipHostMallocMapped
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define hapiMemoryTypeUnregistered hipMemoryTypeUnregistered
#define hapiMemoryTypeHost hipMemoryTypeHost
#define hapiMemoryTypeDevice hipMemoryTypeDevice
#define hapiMemoryTypeManaged hipMemoryTypeManaged
#define hapiDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define hapiDevAttrMaxBlocksPerMultiprocessor hipDeviceAttributeMaxBlocksPerMultiProcessor
#define hapiDevAttrClockRate hipDeviceAttributeClockRate
#define hapiDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define hapiDevAttrMaxRegistersPerMultiprocessor hipDeviceAttributeMaxRegistersPerMultiprocessor
#define hapiDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define hapiDevAttrWarpSize hipDeviceAttributeWarpSize
#define hapiDrvSuccess hipSuccess
#define hapiDrvErrorNotSupported hipErrorNotSupported
#define hapiStreamWriteValueDefault 0u
#define hapiMemHandleTypePosixFd hipMemHandleTypePosixFileDescriptor
#define hapiMemHandleTypeNone hipMemHandleTypeNone
#define hapiMemLocationTypeDevice hipMemLocationTypeDevice
#define hapiMemAllocationTypePinned hipMemAllocationTypePinned
#define hapiMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#define hapiMemAllocGranularityMinimum hipMemAllocationGranularityMinimum
#define hapiMemAllocGranularityRecommended hipMemAllocationGranularityRecommended
// ROCm's count of hardware queues a process's streams are spread over.
#define HAPI_HW_QUEUES_ENV "GPU_MAX_HW_QUEUES"
#define HAPI_HW_QUEUES_DRIVER_DEFAULT "4"
#define HAPI_VENDOR_NAME "HIP"

// ---- runtime API ----
#define hapiGetLastError hipGetLastError
#define hapiPeekAtLastError hipPeekAtLastError
#define hapiGetErrorName hipGetErrorName
#define hapiGetErrorString(err) hipGetErrorString(err)
#define hapiErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define hapiSetDevice(dev) hipSetDevice(dev)
#define hapiGetDevice(dev) hipGetDevice(dev)
#define hapiGetDeviceCount(devCount) hipGetDeviceCount(devCount)
#define hapiGetDeviceProperties(prop, dev) hipGetDeviceProperties(prop, dev)
#define hapiDeviceGetAttribute hipDeviceGetAttribute
#define hapiDeviceSynchronize hipDeviceSynchronize
#define hapiDeviceCanAccessPeer(canAccess, dev1, dev2) \
    hipDeviceCanAccessPeer(canAccess, dev1, dev2)
#define hapiDeviceEnablePeerAccess(dev, flags) \
    hipDeviceEnablePeerAccess(dev, flags)
#define hapiDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define hapiMemGetInfo hipMemGetInfo
#define hapiMallocRaw(ptr, size) hipMalloc((void**)(ptr), size)
#define hapiFreeRaw(ptr) hipFree(ptr)
#define hapiMallocHost(ptr, size) hipHostMalloc((void**)(ptr), size, hipHostMallocDefault)
#define hapiFreeHost(ptr) hipHostFree(ptr)
#define hapiHostAlloc(ptr, size, flags) hipHostMalloc((void**)(ptr), size, flags)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    hipHostGetDevicePointer(devPtr, hostPtr, flags)
#define hapiMemcpy(dst, src, count, kind) hipMemcpy(dst, src, count, kind)
#define hapiMemcpyAsync hipMemcpyAsync
#define hapiMemcpy2DAsync hipMemcpy2DAsync
#define hapiMemsetAsync hipMemsetAsync
#define hapiMemcpyToSymbol(sym, src, n) hipMemcpyToSymbol(HIP_SYMBOL(sym), src, n)
#define hapiStreamCreate(stream) hipStreamCreate(stream)
#define hapiStreamCreateNonBlocking(stream) \
    hipStreamCreateWithFlags(stream, hipStreamNonBlocking)
#define hapiStreamCreateWithFlags hipStreamCreateWithFlags
#define hapiStreamCreateWithPriority hipStreamCreateWithPriority
#define hapiStreamDestroy hipStreamDestroy
#define hapiStreamSynchronize(stream) hipStreamSynchronize(stream)
#define hapiStreamQuery hipStreamQuery
#define hapiEventCreate hipEventCreate
#define hapiEventCreateWithFlags(flags, event) hipEventCreateWithFlags(flags, event)
#define hapiEventQuery(event) hipEventQuery(event)
#define hapiEventSynchronize hipEventSynchronize
#define hapiEventDestroy(event) hipEventDestroy(event)
#define hapiEventElapsedTime(ms, start, end) hipEventElapsedTime(ms, start, end)
#define hapiMemset(ptr, value, count) hipMemset(ptr, value, count)
#define hapiLaunchHostFunc(stream, func, args) \
    hipLaunchHostFunc(stream, func, args)
#define hapiLaunchKernelRaw hipLaunchKernel
#define hapiIpcGetMemHandle(handle, ptr) hipIpcGetMemHandle(handle, ptr)
#define hapiIpcCloseMemHandle(handle) hipIpcCloseMemHandle(handle)
#define hapiIpcGetEventHandle(handle, event) hipIpcGetEventHandle(handle, event)
#define hapiIpcOpenMemHandle(ptr, handle, flags) \
    hipIpcOpenMemHandle(ptr, handle, flags)
#define hapiIpcOpenEventHandle(event, handle) \
    hipIpcOpenEventHandle(event, handle)
#define hapiPointerGetAttributes hipPointerGetAttributes
#define hapiMallocAsyncRaw hipMallocAsync
#define hapiFreeAsyncRaw hipFreeAsync
#define hapiEventRecordRaw hipEventRecord
#define hapiStreamWaitEventRaw hipStreamWaitEvent

// ---- driver level ----
#define HAPI_DEVPTR(x) ((hipDeviceptr_t)(uintptr_t)(x))
static inline hapiDrvResult hapiDrvGetErrorName(hapiDrvResult r, const char** s) { return hipDrvGetErrorName(r, s); }
static inline hapiDrvResult hapiDrvDeviceGet(hapiDrvDevice* d, int ordinal) { return hipDeviceGet(d, ordinal); }
static inline hapiDrvResult hapiDrvDeviceTotalMem(size_t* bytes, hapiDrvDevice d) { return hipDeviceTotalMem(bytes, d); }
// HIP has no attribute for the file-descriptor handle type or for RDMA on
// virtual-memory chunks; both are requested and the pool's first-chunk
// step-down (hapi_impl.cpp) keeps what the runtime accepts.
static inline void hapiDrvDeviceVmmSupport(hapiDrvDevice d, int* vmm, int* fd, int* rdma) {
  *vmm = 0; *fd = 1; *rdma = 1;
  if (hipDeviceGetAttribute(vmm, hipDeviceAttributeVirtualMemoryManagementSupported, (int)d) != hipSuccess) *vmm = 0;
}
static inline hapiDrvResult hapiDrvMemAddressReserve(uintptr_t* out, size_t size, size_t align,
                                                      uintptr_t hint, unsigned long long flags) {
  void* p = nullptr;
  const hipError_t r = hipMemAddressReserve(&p, size, align, (void*)hint, flags);
  *out = (uintptr_t)p;
  return r;
}
static inline hapiDrvResult hapiDrvMemAddressFree(uintptr_t at, size_t size) { return hipMemAddressFree((void*)at, size); }
static inline hapiDrvResult hapiDrvMemCreate(hapiMemGenericAllocationHandle* h, size_t size,
                                             const hapiMemAllocationProp* prop, unsigned long long flags) {
  return hipMemCreate(h, size, prop, flags);
}
static inline hapiDrvResult hapiDrvMemRelease(hapiMemGenericAllocationHandle h) { return hipMemRelease(h); }
static inline hapiDrvResult hapiDrvMemMap(uintptr_t at, size_t size, size_t offset,
                                          hapiMemGenericAllocationHandle h, unsigned long long flags) {
  return hipMemMap((void*)at, size, offset, h, flags);
}
static inline hapiDrvResult hapiDrvMemUnmap(uintptr_t at, size_t size) { return hipMemUnmap((void*)at, size); }
static inline hapiDrvResult hapiDrvMemSetAccess(uintptr_t at, size_t size, const hapiMemAccessDesc* d, size_t n) {
  return hipMemSetAccess((void*)at, size, d, n);
}
static inline hapiDrvResult hapiDrvMemExportToShareableHandle(void* out, hapiMemGenericAllocationHandle h,
                                                              hapiMemAllocationHandleType t, unsigned long long flags) {
  return hipMemExportToShareableHandle(out, h, t, flags);
}
static inline hapiDrvResult hapiDrvMemImportFromShareableHandle(hapiMemGenericAllocationHandle* h, void* os,
                                                                hapiMemAllocationHandleType t) {
  return hipMemImportFromShareableHandle(h, os, t);
}
static inline hapiDrvResult hapiDrvMemGetAllocationGranularity(size_t* g, const hapiMemAllocationProp* p,
                                                               hapiMemAllocationGranularityFlags f) {
  return hipMemGetAllocationGranularity(g, p, f);
}

#endif // CMK_HIP

#if CMK_CUDA || CMK_HIP

// Records and stream waits go through hapi so that +gpustalltrace can log
// them: a stalled stream is then walked back to the wait that never came,
// and the record it was waiting for. Off, these are the plain vendor calls
// behind one predictable branch. The noted forms carry a caller's tag (the
// IPC event slot, say) so a wait in one process can be matched to the
// record in another.
#ifdef __cplusplus
extern "C++" {
hapiError_t hapiEventRecordNoted(hapiEvent_t event, hapiStream_t stream, int note);
hapiError_t hapiStreamWaitEventNoted(hapiStream_t stream, hapiEvent_t event,
                                     unsigned int flags, int note);
}
#define hapiEventRecord(event, stream) hapiEventRecordNoted(event, stream, 0)
#define hapiStreamWaitEvent(stream, event, flags) \
    hapiStreamWaitEventNoted(stream, event, flags, 0)
#else
#if CMK_CUDA
#define hapiEventRecord(event, stream) cudaEventRecord(event, stream)
#define hapiStreamWaitEvent(stream, event, flags) cudaStreamWaitEvent(stream, event, flags)
#else
#define hapiEventRecord(event, stream) hipEventRecord(event, stream)
#define hapiStreamWaitEvent(stream, event, flags) hipStreamWaitEvent(stream, event, flags)
#endif
#endif

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
// call) and hapiFree returns one. Without the pool, the vendor's malloc/free
// as ever. A pointer that predates the pool is still freed with the vendor's
// free. One difference to know: a vendor free waits for the device, a pool
// free does not -- use hapiFreeAsync(ptr, stream) to order the free behind a
// stream's work. Implemented in hapi_impl.cpp.
hapiError_t hapiMallocImpl(void** ptr, size_t size);
hapiError_t hapiFreeImpl(void* ptr);
hapiError_t hapiFreeAsync(void* ptr, hapiStream_t stream);

template <typename hapiMallocT>
static inline hapiError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  void* p = nullptr;
  const hapiError_t hapi_malloc_err = hapiMallocImpl(&p, size);
  if (hapi_malloc_err == hapiSuccess) *ptr = (hapiMallocT*)p;
  return hapi_malloc_err;
}
static inline hapiError_t hapiFreeRecord(void* ptr) { return hapiFreeImpl(ptr); }

#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeRecord(ptr)
// An allocation that belongs to the runtime whatever is running: an arena the
// pool or a comm buffer carves up, whose blocks are charged as they are handed
// out. Charging the arena too would bill a chare for all of it.
#define hapiMallocUnattributed(ptr, size) hapiMallocRaw(ptr, size)

#endif // CMK_CUDA || CMK_HIP

#endif // HAPI_PORTABLE_H_SEEN
