#pragma once

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
#define hapiDevAttrClockRate hipDeviceAttributeClockRate
#define hapiDeviceGetAttribute(a,b,c) hipDeviceGetAttribute(a,b,c)

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

#define hapiMalloc(ptr, size) hipMalloc(ptr, size)
#define hapiFree(ptr) hipFree(ptr)
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
#define hapiEventDefault hipEventDefault
#define hapiEventInterprocess hipEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

#define hapiMemcpyAsync hipMemcpyAsync
#define hapiMemcpy2DAsync hipMemcpy2DAsync

#endif // CMK_HIP
