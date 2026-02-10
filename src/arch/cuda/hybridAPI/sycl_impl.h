#ifndef SYCL_IMPL_H
#define SYCL_IMPL_H

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

// SYCL uses queues instead of streams
#define hapiStream_t sycl::queue*
#define hapiDeviceProp sycl::device

// Error codes
#define hapiErrorMemoryAllocation -1
#define hapiErrorInitializationError -2
#define hapiErrorNotReady -3
#define hapiSuccess 0
#define hapiError_t int

// Memory copy operations
#define hapiMemcpyKind int
#define hapiMemcpyHostToHost 0
#define hapiMemcpyHostToDevice 1
#define hapiMemcpyDeviceToHost 2
#define hapiMemcpyDeviceToDevice 3

// Event flags
#define hapiEventDisableTiming 0x01
#define hapiEventInterprocess 0x02
#define hapiIpcMemLazyEnablePeerAccess 0x01

#define hapiGetErrorString(err) \
    ((err) == hapiSuccess ? "Success" : "Error")

#define hapiStreamPerThread (CpvAccess(sycl_per_thread_stream))

CpvDeclare(hapiStream_t, sycl_per_thread_stream);

// Event structure
typedef struct {
    sycl::event ev;
    int flag;
}* hapiEvent_t;

// IPC Handles - Level Zero based
typedef struct {
    ze_ipc_mem_handle_t ze_handle;
} hapiIpcMemHandle_t;

typedef struct {
    ze_ipc_event_pool_handle_t ze_handle;
} hapiIpcEventHandle_t;

// Function declarations
int hapiSetDevice(int dev);
int hapiStreamCreate(sycl::queue** stream);
int hapiDeviceCanAccessPeer(int* canAccess, int devIdx1, int devIdx2);
int hapiIpcGetMemHandle(hapiIpcMemHandle_t* handle, void* ptr);
int hapiIpcGetEventHandle(hapiIpcEventHandle_t* handle, hapiEvent_t event);
int hapiIpcOpenMemHandle(void** ptr, hapiIpcMemHandle_t handle, int flags);
int hapiGetDevice(int* dev);

// Base hapiMalloc function
int hapiMallocImpl(void** ptr, size_t size);

// Template wrapper for type-safe hapiMalloc
template<typename T>
inline int hapiMalloc(T** ptr, size_t size) {
    return hapiMallocImpl(reinterpret_cast<void**>(ptr), size);
}

int hapiFree(void* ptr);

// Base hapiMallocHost function
int hapiMallocHostImpl(void** ptr, size_t size);

// Template wrapper for type-safe hapiMallocHost
template<typename T>
inline int hapiMallocHost(T** ptr, size_t size) {
    return hapiMallocHostImpl(reinterpret_cast<void**>(ptr), size);
}

int hapiFreeHost(void* ptr);
int hapiMemcpy(void* dst, const void* src, size_t size, hapiMemcpyKind kind);
int hapiMemcpyAsync(void* dst, const void* src, size_t size, hapiMemcpyKind kind, hapiStream_t stream);
int hapiEventCreateWithFlags(hapiEvent_t* event, unsigned int flags);
int hapiEventRecord(hapiEvent_t event, hapiStream_t stream);
int hapiEventQuery(hapiEvent_t event);
int hapiEventDestroy(hapiEvent_t event);
int hapiStreamSynchronize(hapiStream_t stream);
int hapiStreamWaitEvent(hapiStream_t stream, hapiEvent_t event, unsigned int flags);
int hapiStreamDestroy(hapiStream_t stream);
int hapiLaunchHostFunc(hapiStream_t stream, void (*func)(void*), void* args);
int hapiGetDeviceProperties(hapiDeviceProp* prop, int dev);
int hapiIpcOpenEventHandle(hapiEvent_t* event, hapiIpcEventHandle_t handle);
int hapiGetDeviceCount(int* count);
int hapiDeviceEnablePeerAccess(int dev, int flags);

int getNumStreams(hapiDeviceProp& device_prop);

#endif // SYCL_IMPL_H