#ifndef __HAPI_H_
#define __HAPI_H_
#include "hapi_portable.h"

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
  void (*runKernel)(struct hapiWorkRequest* wr, hapiStream_t kernel_stream,
                    void** device_buffers);

  // flag used for control by the system
  int state;

  // may be used to pass data to kernel calls
  void* user_data;

  // flags determining whether memory should be freed on destruction
  bool free_user_data;

  // CUDA stream index provided by the user or assigned by GPUManager
  hapiStream_t stream;

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

  void setRunKernel(void (*_runKernel)(struct hapiWorkRequest*, hapiStream_t, void**)) {
    runKernel = _runKernel;
  }

  void setStream(hapiStream_t _stream) {
    stream = _stream;
  }

  hapiStream_t getStream() {
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
void hapiAddCallback(hapiStream_t stream, const CkCallback& cb, void* cb_msg);
static inline void hapiAddCallback(hapiStream_t stream, const CkCallback& cb) {
  hapiAddCallback(stream, cb, nullptr);
}
static inline void hapiAddCallback(hapiStream_t stream, void* cb) {
  hapiAddCallback(stream, cb, nullptr);
}

// Overloaded C++ wrappers for selecting whether to pool or not using a bool.
static inline hapiError_t hapiMallocHost_Pool(void** ptr, size_t size, bool pool) {
  return pool ? hapiPoolMalloc(ptr, size) : hapiMallocHost(ptr, size);
}
static inline hapiError_t hapiFreeHost_Pool(void* ptr, bool pool) {
  return pool ? hapiPoolFree(ptr) : hapiFreeHost(ptr);
}

#ifdef CMK_LBDB_ON
void hapiCuptiInit();
void hapiCuptiFinalize();
uint64_t hapiCuptiPushObjCorrelation();
void hapiCuptiPopObjCorrelation();
void hapiProcessCuptiBuffers();
void hapiNormalizeCuptiLoads();
void hapiClearCuptiData();

// Start/stop CUPTI activity tracing. Tracing is the dominant cost of GPU load
// instrumentation, and the loads it produces are only read at a load-balancing
// step, so an application that balances on an explicit schedule can leave it
// off and switch it on for a few steps ahead of AtSync. These follow the
// existing LBTurnInstrumentOn()/LBTurnInstrumentOff() switch, so applications
// control GPU tracing with the same call that controls CPU instrumentation.
// Both are idempotent.
void hapiCuptiStartTracing();
void hapiCuptiStopTracing();
bool hapiCuptiTracingActive();
#endif

#ifdef CMK_LBDB_ON
#define CUPTI_LAUNCH_WRAPPER(call)\
  hapiCuptiPushObjCorrelation();\
  call;\
  hapiCuptiPopObjCorrelation();
#else
#define CUPTI_LAUNCH_WRAPPER(call)\
  call;
#endif

// Convenience form of kernel<<<grid, block, shmem, stream>>>(args...).
//
// This carries no instrumentation. CUPTI external-correlation IDs are pushed
// once per entry method in CkCallstackPush/Pop, which covers every kernel
// launched from application code regardless of how it is launched, so there is
// nothing for a per-launch wrapper to add.
//
// Only visible to the CUDA compiler: the <<<>>> launch syntax is not valid C++
// for a host-only compiler, and hapi.h is included by both.
#if defined(__CUDACC__)
template <typename Kernel, typename... Args>
inline void hapiLaunchKernelWrapper(Kernel kernel, dim3 grid_dim, dim3 block_dim,
                                    size_t shared_mem, cudaStream_t stream,
                                    Args... args) {
  kernel<<<grid_dim, block_dim, shared_mem, stream>>>(args...);
}
#endif // __CUDACC__

#endif /* defined __cplusplus */

#endif /* !defined AMPI_INTERNAL_SKIP_FUNCTIONS */

#endif // __HAPI_H_
