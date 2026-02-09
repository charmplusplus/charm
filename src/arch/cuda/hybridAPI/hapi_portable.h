#pragma once

#undef CMK_CUDA
#undef CMK_HIP
#undef CMK_SYCL

#include "conv-mach-opt.h"

#ifdef CMK_CUDA

#include <cuda_runtime.h>

#define hapiStream_t cudaStream_t

#define hapiEvent_t cudaEvent_t

#define hapiSetDevice(dev) cudaSetDevice(dev)

#define hapiPeekAtLastError cudaPeekAtLastError

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
#define hapiMemcpyAsync(dst, src, count, kind, stream) \
    cudaMemcpyAsync(dst, src, count, kind, stream)

#define hapiGetErrorString(err) cudaGetErrorString(err)

#define hapiEventDisableTiming cudaEventDisableTiming
#define hapiEventInterprocess cudaEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

int getNumStreams(hapiDeviceProp& device_prop) {
    int new_n_streams = 0;
    if (device_prop.major == 3) {
      if (device_prop.minor == 0)
        new_n_streams = 16;
      else if (device_prop.minor == 2)
        new_n_streams = 4;
      else // 3.5, 3.7 or unknown 3.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 5) {
      if (device_prop.minor == 3)
        new_n_streams = 16;
      else // 5.0, 5.2 or unknown 5.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 6) {
      if (device_prop.minor == 1)
        new_n_streams = 32;
      else if (device_prop.minor == 2)
        new_n_streams = 16;
      else // 6.0 or unknown 6.x
        new_n_streams = 128;
    }
    else // unknown (future) compute capability
      new_n_streams = 128;
    return new_n_streams;
}

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
#define hapiMemcpyAsync(dst, src, count, kind, stream) \
    hipMemcpyAsync(dst, src, count, kind, stream)
#define hapiGetErrorString(err) hipGetErrorString(err)

#define hapiEventDisableTiming hipEventDisableTiming
#define hapiEventInterprocess hipEventInterprocess
#define hapiIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

int getNumStreams(hapiDeviceProp& device_prop) {
    int new_n_streams = 0;
    if (device_prop.major == 3) {
      if (device_prop.minor == 0)
        new_n_streams = 16;
      else if (device_prop.minor == 2)
        new_n_streams = 4;
      else // 3.5, 3.7 or unknown 3.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 5) {
      if (device_prop.minor == 3)
        new_n_streams = 16;
      else // 5.0, 5.2 or unknown 5.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 6) {
      if (device_prop.minor == 1)
        new_n_streams = 32;
      else if (device_prop.minor == 2)
        new_n_streams = 16;
      else // 6.0 or unknown 6.x
        new_n_streams = 128;
    }
    else // unknown (future) compute capability
      new_n_streams = 128;
    return new_n_streams;
}

#endif // CMK_HIP

#ifdef CMK_SYCL

#include "sycl_impl.h"

#endif // CMK_SYCL
