#ifndef __HAPI_H_
#define __HAPI_H_
#include <cuda_runtime.h>

/* See hapi_functions.h for the majority of function declarations provided
 * by the Hybrid API. */

#ifdef __cplusplus

#include "ckcallback.h"
#include <cstring>
#include <cstdlib>
#include <vector>

/******************** DEPRECATED ********************/
// Contains information about a device buffer, which is used by
// the runtime to perform appropriate operations. Each hapiBufferInfo should
// be associated with a hapiWorkRequest.
typedef struct hapiBufferInfo {
  // ID of buffer in the runtime system's buffer table
  int id;

  // flags to indicate if the buffer should be transferred
  bool transfer_to_device;
  bool transfer_to_host;

  // flag to indicate if the device buffer memory should be freed
  // after execution of work request
  bool need_free;

  // pointer to host data buffer
  void* host_buffer;

  // size of buffer in bytes
  size_t size;

  hapiBufferInfo(int _id = -1) : id(_id), transfer_to_device(false),
    transfer_to_host(false) {}

  hapiBufferInfo(void* _host_buffer, size_t _size, bool _transfer_to_device,
      bool _transfer_to_host, bool _need_free, int _id = -1) :
    host_buffer(_host_buffer), size(_size), transfer_to_device(_transfer_to_device),
    transfer_to_host(_transfer_to_host), need_free(_need_free), id(_id) {}

} hapiBufferInfo;

/******************** DEPRECATED ********************/
// Data structure that ties a kernel, associated buffers, and other variables
// required by the runtime. The user gets a hapiWorkRequest from the runtime,
// fills it in, and enqueues it. The memory associated with it is managed
// by the runtime.
typedef struct hapiWorkRequest {
  // parameters for kernel execution
  dim3 grid_dim;
  dim3 block_dim;
  int shared_mem;

  // contains information about buffers associated with the kernel
  std::vector<hapiBufferInfo> buffers;

  // Charm++ callback functions to be executed after certain stages of
  // GPU execution
  CkCallback host_to_device_cb; // after host to device data transfer
  CkCallback kernel_cb; // after kernel execution
  CkCallback device_to_host_cb; // after device to host data transfer

  bool host_to_device_cb_set;
  bool kernel_cb_set;
  bool device_to_host_cb_set;

#ifdef HAPI_TRACE
  // short identifier used for tracing and logging
  const char *trace_name;
#endif

  // Pointer to host-side function that actually invokes the kernel.
  // The user implements this function, using the given CUDA stream and
  // device buffers (which are indexed by hapiBufferInfo->id).
  // Could be set to NULL if no kernel needs to be executed.
  void (*runKernel)(struct hapiWorkRequest* wr, cudaStream_t kernel_stream,
                    void** device_buffers);

  // flag used for control by the system
  int state;

  // may be used to pass data to kernel calls
  void* user_data;

  // flags determining whether memory should be freed on destruction
  bool free_user_data;

  // CUDA stream index provided by the user or assigned by GPUManager
  cudaStream_t stream;

#ifdef HAPI_INSTRUMENT_WRS
  double phase_start_time;
  int chare_index;
  char comp_type;
  char comp_phase;
#endif

  hapiWorkRequest();

  ~hapiWorkRequest() {
    if (free_user_data)
      std::free(user_data);
  }

  void setExecParams(dim3 _grid_dim, dim3 _block_dim, int _shared_mem = 0) {
    grid_dim = _grid_dim;
    block_dim = _block_dim;
    shared_mem = _shared_mem;
  }

  void addBuffer(void *host_buffer, size_t size, bool transfer_to_device,
                 bool transfer_to_host, bool need_free, int id = -1) {
    buffers.emplace_back(host_buffer, size, transfer_to_device, transfer_to_host,
                         need_free, id);
  }

  int getBufferID(int i) {
    return buffers[i].id;
  }

  int getBufferCount() {
    return buffers.size();
  }

  void setHostToDeviceCallback(const CkCallback& cb) {
    host_to_device_cb = cb;
    host_to_device_cb_set = true;
  }

  void setKernelCallback(const CkCallback& cb) {
    kernel_cb = cb;
    kernel_cb_set = true;
  }

  void setDeviceToHostCallback(const CkCallback& cb) {
    device_to_host_cb = cb;
    device_to_host_cb_set = true;
  }

  inline void setCallback(const CkCallback& cb) {
    setDeviceToHostCallback(cb);
  }

#ifdef HAPI_TRACE
  void setTraceName(const char* _trace_name) {
    trace_name = _trace_name;
  }
#endif

  void setRunKernel(void (*_runKernel)(struct hapiWorkRequest*, cudaStream_t, void**)) {
    runKernel = _runKernel;
  }

  void setStream(cudaStream_t _stream) {
    stream = _stream;
  }

  cudaStream_t getStream() {
    return stream;
  }

  void copyUserData(void* ptr, size_t size) {
    // make a separate copy to prevent tampering with the original data
    free_user_data = true;
    user_data = std::malloc(size);
    std::memcpy(user_data, ptr, size);
  }

  void setUserData(void* ptr, bool _free_user_data = false) {
    free_user_data = _free_user_data;
    user_data = ptr;
  }

  void* getUserData() {
    return user_data;
  }

} hapiWorkRequest;

#else /* defined __cplusplus */

/* In C mode, only declare the existence of C++ structs. */
typedef struct hapiBufferInfo hapiBufferInfo;
typedef struct hapiWorkRequest hapiWorkRequest;

#endif /* defined __cplusplus */

// Provides support for detecting errors with CUDA API calls.
#ifndef HAPI_CHECK_OFF
#define hapiCheck(code) hapiErrorDie(code, #code, __FILE__, __LINE__)
#else
#define hapiCheck(code) code
#endif

#ifdef HAPI_INSTRUMENT_WRS
typedef struct hapiRequestTimeInfo {
  double transfer_time;
  double kernel_time;
  double cleanup_time;
  int n;

#ifdef __cplusplus
  hapiRequestTimeInfo() : transfer_time(0.0), kernel_time(0.0), cleanup_time(0.0),
    n(0) {}
#endif /* defined __cplusplus */
} hapiRequestTimeInfo;
#endif /* defined HAPI_INSTRUMENT_WRS */

#ifndef AMPI_INTERNAL_SKIP_FUNCTIONS

#define AMPI_CUSTOM_FUNC(return_type, function_name, ...) \
extern return_type function_name(__VA_ARGS__);

#ifdef __cplusplus
extern "C" {
#endif
#include "hapi_functions.h"
#ifdef __cplusplus
}
#endif

#undef AMPI_CUSTOM_FUNC

#ifdef __cplusplus

// Provide a C++-only stub for this function's default parameter.
void hapiAddCallback(cudaStream_t stream, const CkCallback& cb, void* cb_msg);
static inline void hapiAddCallback(cudaStream_t stream, const CkCallback& cb) {
  hapiAddCallback(stream, cb, nullptr);
}
static inline void hapiAddCallback(cudaStream_t stream, void* cb) {
  hapiAddCallback(stream, cb, nullptr);
}

// Overloaded C++ wrappers for selecting whether to pool or not using a bool.
static inline cudaError_t hapiMallocHost(void** ptr, size_t size, bool pool) {
  return pool ? hapiPoolMalloc(ptr, size) : hapiMallocHost(ptr, size);
}
static inline cudaError_t hapiFreeHost(void* ptr, bool pool) {
  return pool ? hapiPoolFree(ptr) : hapiFreeHost(ptr);
}

// C++ template wrapper for easier kernel launching with timing
template<typename... Args>
static inline cudaError_t hapiLaunchKernelWrapper(void (*kernel)(Args...), dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, Args... args) {
  void* kernel_args[] = {(void*)&args...};
  return hapiLaunchKernel((const void*)kernel, gridDim, blockDim, kernel_args, sharedMem, stream);
}

// Overload for kernels with no arguments
static inline cudaError_t hapiLaunchKernelWrapper(void (*kernel)(), dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
  return hapiLaunchKernel((const void*)kernel, gridDim, blockDim, nullptr, sharedMem, stream);
}

#endif /* defined __cplusplus */

#endif /* !defined AMPI_INTERNAL_SKIP_FUNCTIONS */

#endif // __HAPI_H_
