/* A named include guard, not #pragma once: the build installs a copy of this
 * header into include/, so a translation unit that includes both the in-tree
 * copy and the installed one (hapi_impl.cpp reaches the latter through
 * charm++.h -> ckrdmadevice.h -> conv-rdmadevice.h) sees two distinct files
 * and #pragma once guards neither against the other. That was harmless while
 * this header held only macros -- identical macro redefinition is legal --
 * but not once it defines functions. */
#ifndef __HAPI_PORTABLE_H_
#define __HAPI_PORTABLE_H_

/* Everything below is C++ even when the includer is not. ampi.h wraps its
 * includes in extern "C", and reaches this header through ampi_functions.h ->
 * hapi_functions.h; without this, the template overloads further down are
 * rejected outright ("template with C linkage"), and on the HIP side
 * hip_runtime.h's own <thread> include fails the same way before we get that
 * far. extern "C++" restores C++ linkage for the whole body regardless of the
 * context the header lands in. */
#ifdef __cplusplus
extern "C++" {
#endif

#undef CMK_CUDA
#undef CMK_HIP

#include "conv-mach-opt.h"
#include <stdint.h>
#include <stddef.h>

#ifdef CMK_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#define hapiStream_t cudaStream_t

#define hapiEvent_t cudaEvent_t

#define hapiSetDevice(dev) cudaSetDevice(dev)

#define hapiDevAttrClockRate cudaDevAttrClockRate
#define hapiDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define hapiDeviceGetAttribute(a,b,c) cudaDeviceGetAttribute(a,b,c)

/* Added in CUDA 11.0. Gated on the runtime version, not #ifdef on the name
 * itself: these are enumerators of cudaDeviceAttr, not macros, so an #ifdef
 * test against one is always false. HAPI_HAS_MAX_BLOCKS_PER_SM is how a
 * caller asks whether the attribute is spellable in this build. */
#if CUDART_VERSION >= 11000
#define HAPI_HAS_MAX_BLOCKS_PER_SM 1
#define hapiDevAttrMaxBlocksPerMultiprocessor \
    cudaDevAttrMaxBlocksPerMultiprocessor
#else
#define HAPI_HAS_MAX_BLOCKS_PER_SM 0
#endif

#define hapiPeekAtLastError cudaPeekAtLastError
#define hapiGetLastError cudaGetLastError
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
#define hapiStreamQuery(stream) cudaStreamQuery(stream)
#define hapiDeviceSynchronize cudaDeviceSynchronize
#define hapiEventElapsedTime(a, b, c) cudaEventElapsedTime(a, b, c)
#define hapiMemGetInfo(a, b) cudaMemGetInfo(a, b)
#define hapiStreamCreate(stream) cudaStreamCreate(stream)
#define hapiStreamDestroy cudaStreamDestroy
#define hapiStreamDefault cudaStreamDefault
#define hapiStreamNonBlocking cudaStreamNonBlocking
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

/* The vendor's allocator itself. hapiMalloc / hapiFree, defined at the end of
 * this header, take memory from the device pool under +gpupool and charge it
 * to the running chare. */
#define hapiMallocRaw(ptr, size) cudaMalloc((void**)(ptr), size)
#define hapiFreeRaw(ptr) cudaFree(ptr)
/* hapiMallocHost/hapiFreeHost are real functions, not macros, in C++.  The
 * pooled forms in hapi.h -- hapiMallocHost(ptr, size, pool) and
 * hapiFreeHost(ptr, pool) -- are overloads of these names, and a function-like
 * macro cannot be overloaded: with a macro here, every 3-argument call becomes
 * a hard preprocessor error ("macro passed 3 arguments, but takes just 2").
 * That signature is part of the public HAPI surface -- ChaNGa calls it -- so
 * the macro form is kept only for the C path, which has no overloads anyway. */
#ifdef __cplusplus
static inline cudaError_t hapiMallocHost(void** ptr, size_t size) {
  return cudaMallocHost(ptr, size);
}
/* cudaMallocHost also declares a C++ template overload on T**, so user code
 * has never needed a cast: hapiMallocHost(&d_ptr, n) with a double** just
 * works. Without this overload the void** signature above is a narrowing of
 * the API this header stands in for, and every such call stops compiling --
 * which is what happened to the gpudirect examples. */
template <typename T>
static inline cudaError_t hapiMallocHost(T** ptr, size_t size) {
  return cudaMallocHost(reinterpret_cast<void**>(ptr), size);
}
static inline cudaError_t hapiFreeHost(void* ptr) {
  return cudaFreeHost(ptr);
}
#else
#define hapiMallocHost(ptr, size) cudaMallocHost(ptr, size)
#define hapiFreeHost(ptr) cudaFreeHost(ptr)
#endif

#define hapiErrorMemoryAllocation cudaErrorMemoryAllocation
#define hapiErrorInitializationError cudaErrorInitializationError
#define hapiErrorNotReady cudaErrorNotReady
#define hapiErrorNotSupported cudaErrorNotSupported
#define hapiGetErrorName cudaGetErrorName
#define hapiSuccess cudaSuccess
#define hapiError_t cudaError_t

/* Pinned-flag completion (hapi_impl.cpp): a mapped pinned host ring, written
 * from a stream by a driver-level call (looked up at startup, so libcuda need
 * not be on the link line), and the device a stream belongs to. */
#define hapiDrvResult CUresult
#define hapiDrvStream CUstream
#define hapiDrvSuccess CUDA_SUCCESS
#define hapiStreamWriteValueDefault CU_STREAM_WRITE_VALUE_DEFAULT
#define hapiHostAllocPortable cudaHostAllocPortable
#define hapiHostAllocMapped cudaHostAllocMapped
#define hapiHostAlloc(ptr, size, flags) cudaHostAlloc(ptr, size, flags)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    cudaHostGetDevicePointer(devPtr, hostPtr, flags)
#define hapiStreamCreateWithFlags cudaStreamCreateWithFlags
#if CUDART_VERSION >= 12080
#define HAPI_HAS_STREAM_GET_DEVICE 1
#define hapiStreamGetDevice(stream, dev) cudaStreamGetDevice(stream, dev)
#else
#define HAPI_HAS_STREAM_GET_DEVICE 0
#endif

/* The device pool's virtual-memory backend (+gpupoolalloc vmm, hapi_impl.cpp):
 * a reserved address range backed in place by physical chunks. Driver-level
 * calls, so they take integer device addresses on both vendors. */
#define hapiDrvDevice CUdevice
#define hapiMemGenericAllocationHandle CUmemGenericAllocationHandle
#define hapiMemAllocationProp CUmemAllocationProp
#define hapiMemAccessDesc CUmemAccessDesc
#define hapiMemAllocationGranularityFlags CUmemAllocationGranularity_flags
#define hapiMemHandleTypeNone CU_MEM_HANDLE_TYPE_NONE
#define hapiMemLocationTypeDevice CU_MEM_LOCATION_TYPE_DEVICE
#define hapiMemAllocationTypePinned CU_MEM_ALLOCATION_TYPE_PINNED
#define hapiMemAccessFlagsProtReadWrite CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define hapiMemAllocGranularityMinimum CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define hapiMemAllocGranularityRecommended CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#ifdef __cplusplus
static inline hapiDrvResult hapiDrvGetErrorName(hapiDrvResult r, const char** s) { return cuGetErrorName(r, s); }
static inline hapiDrvResult hapiDrvDeviceGet(hapiDrvDevice* d, int ordinal) { return cuDeviceGet(d, ordinal); }
static inline hapiDrvResult hapiDrvDeviceTotalMem(size_t* bytes, hapiDrvDevice d) { return cuDeviceTotalMem(bytes, d); }
/* Whether the device can back a reserved range with physical chunks, and have
 * those chunks registered by the NIC. */
static inline void hapiDrvDeviceVmmSupport(hapiDrvDevice d, int* vmm, int* rdma) {
  *vmm = *rdma = 0;
  cuDeviceGetAttribute(vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d);
  cuDeviceGetAttribute(rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, d);
}
static inline hapiDrvResult hapiDrvMemAddressReserve(uintptr_t* out, size_t size, size_t align,
                                                      uintptr_t hint, unsigned long long flags) {
  CUdeviceptr p = 0;
  const CUresult r = cuMemAddressReserve(&p, size, align, (CUdeviceptr)hint, flags);
  *out = (uintptr_t)p;
  return r;
}
static inline hapiDrvResult hapiDrvMemCreate(hapiMemGenericAllocationHandle* h, size_t size,
                                             const hapiMemAllocationProp* prop, unsigned long long flags) {
  return cuMemCreate(h, size, prop, flags);
}
static inline hapiDrvResult hapiDrvMemMap(uintptr_t at, size_t size, size_t offset,
                                          hapiMemGenericAllocationHandle h, unsigned long long flags) {
  return cuMemMap((CUdeviceptr)at, size, offset, h, flags);
}
static inline hapiDrvResult hapiDrvMemSetAccess(uintptr_t at, size_t size, const hapiMemAccessDesc* d, size_t n) {
  return cuMemSetAccess((CUdeviceptr)at, size, d, n);
}
static inline hapiDrvResult hapiDrvMemGetAllocationGranularity(size_t* g, const hapiMemAllocationProp* p,
                                                               hapiMemAllocationGranularityFlags f) {
  return cuMemGetAllocationGranularity(g, p, f);
}
#endif

#define hapiMemcpyKind cudaMemcpyKind
#define hapiMemcpyHostToHost cudaMemcpyHostToHost
#define hapiMemcpyHostToDevice cudaMemcpyHostToDevice
#define hapiMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hapiMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hapiMemcpyDefault cudaMemcpyDefault
#define hapiMemcpy(dst, src, count, kind) cudaMemcpy(dst, src, count, kind)
#define hapiMemcpy2D(dst, dpitch, src, spitch, width, height, kind) \
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

#define hapiGetErrorString(err) cudaGetErrorString(err)

#define hapiEventInterprocess cudaEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync cudaMemcpyAsync
#define hapiMemcpy2DAsync cudaMemcpy2DAsync
#define hapiMemset(ptr, value, count) cudaMemset(ptr, value, count)
#define hapiMemsetAsync(ptr, value, count, stream) \
    cudaMemsetAsync(ptr, value, count, stream)

#endif // CMK_CUDA

#ifdef CMK_HIP

// This header is pulled in by charm++.h, so any user code compiled against a
// HIP-enabled build ends up including hip_runtime.h through a plain host
// compiler (g++), which does not set a HIP platform macro on its own. The
// Charm++ build itself gets this from cmake; supply it here so user code
// compiled with charmc sees the same platform.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>

#define hapiStream_t hipStream_t

#define hapiEvent_t hipEvent_t

#define hapiSetDevice(dev) hipSetDevice(dev)
#define hapiGetDeviceCount(devCount) hipGetDeviceCount(devCount)
#define hapiDevAttrClockRate hipDeviceAttributeClockRate
#define hapiDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define hapiDeviceGetAttribute(a,b,c) hipDeviceGetAttribute(a,b,c)

/* See the CUDA side: HAPI_HAS_MAX_BLOCKS_PER_SM says whether the attribute is
 * spellable here. HIP has carried hipDeviceAttributeMaxBlocksPerMultiProcessor
 * since ROCm 4.5; older ROCm is not supported by this build anyway. */
#define HAPI_HAS_MAX_BLOCKS_PER_SM 1
#define hapiDevAttrMaxBlocksPerMultiprocessor \
    hipDeviceAttributeMaxBlocksPerMultiProcessor

#define hapiPeekAtLastError hipPeekAtLastError
#define hapiGetLastError hipGetLastError

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
#define hapiStreamQuery(stream) hipStreamQuery(stream)
#define hapiDeviceSynchronize hipDeviceSynchronize
#define hapiEventElapsedTime(a, b, c) hipEventElapsedTime(a, b, c)
#define hapiMemGetInfo(a, b) hipMemGetInfo(a, b)
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
#define hapiStreamDestroy hipStreamDestroy
#define hapiStreamDefault hipStreamDefault
#define hapiStreamNonBlocking hipStreamNonBlocking
#define hapiStreamCreateWithPriority hipStreamCreateWithPriority

/* The vendor's allocator itself; see the CUDA table. */
#define hapiMallocRaw(ptr, size) hipMalloc((void**)(ptr), size)
#define hapiFreeRaw(ptr) hipFree(ptr)
/* hapiMallocHost/hapiFreeHost are real functions, not macros, in C++.  The
 * pooled forms in hapi.h -- hapiMallocHost(ptr, size, pool) and
 * hapiFreeHost(ptr, pool) -- are overloads of these names, and a function-like
 * macro cannot be overloaded: with a macro here, every 3-argument call becomes
 * a hard preprocessor error ("macro passed 3 arguments, but takes just 2").
 * That signature is part of the public HAPI surface -- ChaNGa calls it -- so
 * the macro form is kept only for the C path, which has no overloads anyway. */
#ifdef __cplusplus
static inline hipError_t hapiMallocHost(void** ptr, size_t size) {
  return hipHostMalloc(ptr, size);
}
/* hipHostMalloc also declares a C++ template overload on T**, so user code
 * has never needed a cast: hapiMallocHost(&d_ptr, n) with a double** just
 * works. Without this overload the void** signature above is a narrowing of
 * the API this header stands in for, and every such call stops compiling --
 * which is what happened to the gpudirect examples. */
template <typename T>
static inline hipError_t hapiMallocHost(T** ptr, size_t size) {
  return hipHostMalloc(reinterpret_cast<void**>(ptr), size);
}
static inline hipError_t hapiFreeHost(void* ptr) {
  return hipHostFree(ptr);
}
#else
/* hipHostMalloc's third parameter is defaulted only in its C++ declaration,
 * so the C spelling has to pass the flag explicitly. hipHostMallocDefault is
 * that same default (0). */
#define hapiMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define hapiFreeHost(ptr) hipHostFree(ptr)
#endif

#define hapiErrorMemoryAllocation hipErrorMemoryAllocation
#define hapiErrorInitializationError hipErrorInitializationError
#define hapiErrorNotReady hipErrorNotReady
#define hapiErrorNotSupported hipErrorNotSupported
#define hapiGetErrorName hipGetErrorName
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
#define hapiMemcpyDefault hipMemcpyDefault
#define hapiMemcpy(dst, src, count, kind) hipMemcpy(dst, src, count, kind)
#define hapiMemcpy2D(dst, dpitch, src, spitch, width, height, kind) \
    hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
#define hapiGetErrorString(err) hipGetErrorString(err)

#define hapiEventDisableTiming hipEventDisableTiming
#define hapiEventDefault hipEventDefault
#define hapiEventInterprocess hipEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync hipMemcpyAsync
#define hapiMemcpy2DAsync hipMemcpy2DAsync
#define hapiMemset(ptr, value, count) hipMemset(ptr, value, count)
#define hapiMemsetAsync(ptr, value, count, stream) \
    hipMemsetAsync(ptr, value, count, stream)

/* Pinned-flag completion (hapi_impl.cpp): a mapped pinned host ring, written
 * from a stream by a driver-level call, and the device a stream belongs to.
 * HIP has the writers as plain runtime calls. */
#define hapiDrvResult hipError_t
#define hapiDrvStream hipStream_t
#define hapiDrvSuccess hipSuccess
#define hapiStreamWriteValueDefault 0u
#define hapiHostAllocPortable hipHostMallocPortable
#define hapiHostAllocMapped hipHostMallocMapped
#define hapiHostAlloc(ptr, size, flags) hipHostMalloc((void**)(ptr), size, flags)
#define hapiHostGetDevicePointer(devPtr, hostPtr, flags) \
    hipHostGetDevicePointer(devPtr, hostPtr, flags)
#define hapiStreamCreateWithFlags hipStreamCreateWithFlags
#define HAPI_HAS_STREAM_GET_DEVICE 1
#define hapiStreamGetDevice(stream, dev) hipStreamGetDevice(stream, dev)

/* The device pool's virtual-memory backend; see the CUDA table. */
#define hapiDrvDevice hipDevice_t
#define hapiMemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define hapiMemAllocationProp hipMemAllocationProp
#define hapiMemAccessDesc hipMemAccessDesc
#define hapiMemAllocationGranularityFlags hipMemAllocationGranularity_flags
#define hapiMemHandleTypeNone hipMemHandleTypeNone
#define hapiMemLocationTypeDevice hipMemLocationTypeDevice
#define hapiMemAllocationTypePinned hipMemAllocationTypePinned
#define hapiMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#define hapiMemAllocGranularityMinimum hipMemAllocationGranularityMinimum
#define hapiMemAllocGranularityRecommended hipMemAllocationGranularityRecommended
#ifdef __cplusplus
static inline hapiDrvResult hapiDrvGetErrorName(hapiDrvResult r, const char** s) { *s = hipGetErrorName(r); return hipSuccess; }
static inline hapiDrvResult hapiDrvDeviceGet(hapiDrvDevice* d, int ordinal) { return hipDeviceGet(d, ordinal); }
static inline hapiDrvResult hapiDrvDeviceTotalMem(size_t* bytes, hapiDrvDevice d) { return hipDeviceTotalMem(bytes, d); }
/* HIP has no attribute for RDMA on virtual-memory chunks: it is requested,
 * and the pool's first-chunk step-down (hapi_impl.cpp) keeps what the runtime
 * accepts. */
static inline void hapiDrvDeviceVmmSupport(hapiDrvDevice d, int* vmm, int* rdma) {
  *vmm = 0; *rdma = 1;
  if (hipDeviceGetAttribute(vmm, hipDeviceAttributeVirtualMemoryManagementSupported, (int)d) != hipSuccess) *vmm = 0;
}
static inline hapiDrvResult hapiDrvMemAddressReserve(uintptr_t* out, size_t size, size_t align,
                                                      uintptr_t hint, unsigned long long flags) {
  void* p = nullptr;
  const hipError_t r = hipMemAddressReserve(&p, size, align, (void*)hint, flags);
  *out = (uintptr_t)p;
  return r;
}
static inline hapiDrvResult hapiDrvMemCreate(hapiMemGenericAllocationHandle* h, size_t size,
                                             const hapiMemAllocationProp* prop, unsigned long long flags) {
  return hipMemCreate(h, size, prop, flags);
}
static inline hapiDrvResult hapiDrvMemMap(uintptr_t at, size_t size, size_t offset,
                                          hapiMemGenericAllocationHandle h, unsigned long long flags) {
  return hipMemMap((void*)at, size, offset, h, flags);
}
static inline hapiDrvResult hapiDrvMemSetAccess(uintptr_t at, size_t size, const hapiMemAccessDesc* d, size_t n) {
  return hipMemSetAccess((void*)at, size, d, n);
}
static inline hapiDrvResult hapiDrvMemGetAllocationGranularity(size_t* g, const hapiMemAllocationProp* p,
                                                               hapiMemAllocationGranularityFlags f) {
  return hipMemGetAllocationGranularity(g, p, f);
}
#endif

#endif // CMK_HIP

#if defined(CMK_CUDA) || defined(CMK_HIP)
/* hapiMalloc / hapiFree. Under +gpupool hapiMalloc hands out a block of the
 * device pool (no driver call) and hapiFree returns one; otherwise they are
 * the vendor's malloc and free. A pointer that is not the pool's is always
 * freed with the vendor's free. A vendor free waits for the device and a pool
 * free does not: hapiFreeAsync(ptr, stream) orders the free of a pool block
 * behind the work queued on `stream`. Implemented in hapi_impl.cpp, which
 * also charges each allocation made inside an entry method to the running
 * chare (the LB memory contract's footprint). The C spelling is the vendor's
 * allocator, uncharged. */
#ifdef __cplusplus
hapiError_t hapiMallocImpl(void** ptr, size_t size);
hapiError_t hapiFreeImpl(void* ptr);
hapiError_t hapiFreeAsync(void* ptr, hapiStream_t stream);
template <typename hapiMallocT>
static inline hapiError_t hapiMallocRecord(hapiMallocT** ptr, size_t size) {
  void* p = nullptr;
  const hapiError_t err = hapiMallocImpl(&p, size);
  if (err == hapiSuccess) *ptr = (hapiMallocT*)p;
  return err;
}
#define hapiMalloc(ptr, size) hapiMallocRecord(ptr, size)
#define hapiFree(ptr) hapiFreeImpl(ptr)
#else
#define hapiMalloc(ptr, size) hapiMallocRaw(ptr, size)
#define hapiFree(ptr) hapiFreeRaw(ptr)
#endif
/* An allocation that belongs to the runtime whatever is running: an arena the
 * pool or a comm buffer carves up, whose blocks are charged as they are
 * handed out. Charging the arena too would bill one chare for all of it. */
#define hapiMallocUnattributed(ptr, size) hapiMallocRaw(ptr, size)
#endif

#ifdef __cplusplus
}  /* extern "C++" */
#endif

#endif /* __HAPI_PORTABLE_H_ */
