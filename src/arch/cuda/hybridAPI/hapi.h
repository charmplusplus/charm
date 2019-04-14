#ifndef __HAPI_H_
#define __HAPI_H_
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>

/******************** DEPRECATED ********************/
// HAPI wrappers whose behavior is controlled by user defined variables,
// which are HAPI_USE_CUDAMALLOCHOST and HAPI_MEMPOOL.
#ifdef HAPI_USE_CUDAMALLOCHOST
#  ifdef HAPI_MEMPOOL
#    define hapiHostMalloc hapiPoolMalloc
#    define hapiHostFree   hapiPoolFree
#  else
#    define hapiHostMalloc cudaMallocHost
#    define hapiHostFree   cudaFreeHost
#  endif // HAPI_MEMPOOL
#else
#  define hapiHostMalloc malloc
#  define hapiHostFree   free
#endif // HAPI_USE_CUDAMALLOCHOST

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
  void* host_to_device_cb; // after host to device data transfer
  void* kernel_cb; // after kernel execution
  void* device_to_host_cb; // after device to host data transfer

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

  // flag determining whether user data is freed on destruction
  bool free_user_data;

  // CUDA stream index provided by the user or assigned by GPUManager
  cudaStream_t stream;

#ifdef HAPI_INSTRUMENT_WRS
  double phase_start_time;
  int chare_index;
  char comp_type;
  char comp_phase;
#endif

  hapiWorkRequest() :
    grid_dim(0), block_dim(0), shared_mem(0), host_to_device_cb(NULL),
    kernel_cb(NULL), device_to_host_cb(NULL), runKernel(NULL), state(0),
    user_data(NULL), free_user_data(false), stream(NULL)
  {
#ifdef HAPI_TRACE
    trace_name = "";
#endif
#ifdef HAPI_INSTRUMENT_WRS
    chare_index = -1;
#endif
  }

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

  void setHostToDeviceCallback(void* cb) {
    host_to_device_cb = cb;
  }

  void setKernelCallback(void* cb) {
    kernel_cb = cb;
  }

  void setDeviceToHostCallback(void* cb) {
    device_to_host_cb = cb;
  }

  void setCallback(void* cb) {
    device_to_host_cb = cb;
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

/******************** DEPRECATED ********************/
// Create a hapiWorkRequest object for the user. The runtime manages the associated
// memory, so the user only needs to set it up properly.
hapiWorkRequest* hapiCreateWorkRequest();

/******************** DEPRECATED ********************/
// Add a work request into the "queue". Currently all specified data transfers
// and kernel execution are directly put into a CUDA stream.
void hapiEnqueue(hapiWorkRequest* wr);

// The runtime queries the compute capability of the device, and creates as
// many streams as the maximum number of concurrent kernels.
int hapiCreateStreams();

// Get a CUDA stream that was created by the runtime. Current scheme is to
// hand out streams in a round-robin fashion.
cudaStream_t hapiGetStream();

// Add a Charm++ callback function to be invoked after the previous operation
// in the stream completes. This call should be placed after data transfers or
// a kernel invocation.
void hapiAddCallback(cudaStream_t, void*, void* = NULL);

// Thin wrappers for memory related CUDA API calls.
cudaError_t hapiMalloc(void**, size_t);
cudaError_t hapiFree(void*);
cudaError_t hapiMallocHost(void**, size_t);
cudaError_t hapiFreeHost(void*);
cudaError_t hapiMallocHostPool(void**, size_t);
cudaError_t hapiFreeHostPool(void*);
#ifdef __cplusplus
// Overloaded versions for C++ code
static inline cudaError_t hapiMallocHost(void** ptr, size_t size, bool pool) {
  return pool ? hapiMallocHostPool(ptr, size) : hapiMallocHost(ptr, size);
}

static inline cudaError_t hapiFreeHost(void* ptr, bool pool) {
  return pool ? hapiFreeHostPool(ptr) : hapiFreeHost(ptr);
}
#endif
cudaError_t hapiMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);

// Explicit memory allocations using pinned memory pool.
void* hapiPoolMalloc(size_t);
void hapiPoolFree(void*);

// Provides support for detecting errors with CUDA API calls.
#ifndef HAPI_CHECK_OFF
#define hapiCheck(code) hapiErrorDie(code, #code, __FILE__, __LINE__)
#else
#define hapiCheck(code) code
#endif
void hapiErrorDie(cudaError_t, const char*, const char*, int);

#ifdef HAPI_INSTRUMENT_WRS
struct hapiRequestTimeInfo {
  double transfer_time;
  double kernel_time;
  double cleanup_time;
  int n;

  hapiRequestTimeInfo() : transfer_time(0.0), kernel_time(0.0), cleanup_time(0.0),
    n(0) {}
};

void hapiInitInstrument(int n_chares, char n_types);
hapiRequestTimeInfo* hapiQueryInstrument(int chare, char type, char phase);
#endif

#endif // __HAPI_H_
