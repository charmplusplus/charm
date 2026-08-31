/* A named include guard, not #pragma once: the build installs a copy of this
 * header into include/, so a translation unit that includes both the in-tree
 * copy and the installed one (hapi_impl.cpp reaches the latter through
 * charm++.h -> ckrdmadevice.h -> conv-rdmadevice.h) sees two distinct files
 * and #pragma once guards neither against the other. That was harmless while
 * this header held only macros -- identical macro redefinition is legal --
 * but not once it defines functions. */
#ifndef __HAPI_PORTABLE_H_
#define __HAPI_PORTABLE_H_

#undef CMK_CUDA
#undef CMK_HIP

#include "conv-mach-opt.h"

#ifdef CMK_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#define hapiStream_t cudaStream_t

#define hapiEvent_t cudaEvent_t

#define hapiSetDevice(dev) cudaSetDevice(dev)

#define hapiDevAttrClockRate cudaDevAttrClockRate
#define hapiDeviceGetAttribute(a,b,c) cudaDeviceGetAttribute(a,b,c)

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

#define hapiMalloc(ptr, size) cudaMalloc(ptr, size)
#define hapiFree(ptr) cudaFree(ptr)
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
static inline cudaError_t hapiFreeHost(void* ptr) {
  return cudaFreeHost(ptr);
}
#else
#define hapiMallocHost(ptr, size) cudaMallocHost(ptr, size)
#define hapiFreeHost(ptr) cudaFreeHost(ptr)
#endif

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
#define hapiMemcpy2D(dst, dpitch, src, spitch, width, height, kind) \
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

#define hapiGetErrorString(err) cudaGetErrorString(err)

#define hapiEventInterprocess cudaEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync cudaMemcpyAsync
#define hapiMemcpy2DAsync cudaMemcpy2DAsync

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
#define hapiDeviceGetAttribute(a,b,c) hipDeviceGetAttribute(a,b,c)

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

#define hapiMalloc(ptr, size) hipMalloc(ptr, size)
#define hapiFree(ptr) hipFree(ptr)
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
#define hapiMemcpy2D(dst, dpitch, src, spitch, width, height, kind) \
    hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
#define hapiGetErrorString(err) hipGetErrorString(err)

#define hapiEventDisableTiming hipEventDisableTiming
#define hapiEventDefault hipEventDefault
#define hapiEventInterprocess hipEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync hipMemcpyAsync
#define hapiMemcpy2DAsync hipMemcpy2DAsync

#endif // CMK_HIP

#endif /* __HAPI_PORTABLE_H_ */
