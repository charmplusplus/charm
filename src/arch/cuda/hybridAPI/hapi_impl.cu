#include "hapi.h"
#include "hapi_impl.h"
#include "converse.h"
#include "ckcallback.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <atomic>

#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
#endif

#if defined HAPI_MEMPOOL || defined HAPI_INSTRUMENT_WRS
#include "cklists.h"
#endif

#if defined HAPI_TRACE || defined HAPI_INSTRUMENT_WRS
extern "C" double CmiWallTimer();
#endif

#ifdef HAPI_TRACE
#define QUEUE_SIZE_INIT 128
extern "C" int traceRegisterUserEvent(const char* x, int e);
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);

typedef struct gpuEventTimer {
  int stage;
  double cmi_start_time;
  double cmi_end_time;
  int event_type;
  const char* trace_name;
} gpuEventTimer;
#endif

#ifdef HAPI_INSTRUMENT_WRS
static bool initializedInstrument();
#endif

// Event stages used for profiling.
enum WorkRequestStage{
  DataSetup        = 1,
  KernelExecution  = 2,
  DataCleanup      = 3
};

enum ProfilingStage{
  GpuMemSetup   = 8800,
  GpuKernelExec = 8801,
  GpuMemCleanup = 8802
};

#ifndef HAPI_CUDA_CALLBACK
typedef struct hapiEvent {
  cudaEvent_t event;
  void* cb;
  void* cb_msg;
} hapiEvent;

// Generate a globally unique pointer to use as a flag
static char free_buff_flag_sentinel;
#define FREE_BUFF_FLAG ((void*)&free_buff_flag_sentinel)

CpvDeclare(std::queue<hapiEvent>, hapi_event_queue);
#endif
CpvDeclare(int, n_hapi_events);

void initEventQueues() {
#ifndef HAPI_CUDA_CALLBACK
  CpvInitialize(std::queue<hapiEvent>, hapi_event_queue);
#endif
  CpvInitialize(int, n_hapi_events);
  CpvAccess(n_hapi_events) = 0;
}

// Returns the CUDA device associated with the given PE.
// TODO: should be updated to exploit the hardware topology instead of round robin
static inline int getMyCudaDevice(int my_pe) {
  int device_count;
  hapiCheck(cudaGetDeviceCount(&device_count));
  return my_pe % device_count;
}

// A function in ck.C which casts the void* to a CkCallback object and invokes
// the Charm++ callback.
extern void CUDACallbackManager(void* fn);
extern int CmiMyPe();

// Functions used to support quiescence detection.
extern void QdCreate(int n);
extern void QdProcess(int n);

// Initial size of the user-addressed portion of host/device buffer arrays;
// the system-addressed portion of host/device buffer arrays (used when there
// is no need to share buffers between work requests) will be equivalant in size.
// FIXME hard-coded maximum
#if CMK_SMP
#define NUM_BUFFERS 4096
#else
#define NUM_BUFFERS 256
#endif

#define MAX_PINNED_REQ 64
#define MAX_DELAYED_FREE_REQS 64

// Contains data and methods needed by HAPI.
class GPUManager {

public:
#ifdef HAPI_MEMPOOL
// Update for new row, again this shouldn't be hard coded!
#define HAPI_MEMPOOL_NUM_SLOTS 20
// Pre-allocated buffers will be at least this big (in bytes).
#define HAPI_MEMPOOL_MIN_BUFFER_SIZE 256
// Scale the amount of memory each node pins.
#define HAPI_MEMPOOL_SCALE 1.0

  CkVec<BufferPool> mempool_free_bufs_;
  CkVec<size_t> mempool_boundaries_;
#endif // HAPI_MEMPOOL

  // The runtime system keeps track of all allocated buffers on the GPU.
  // The following arrays contain pointers to host (CPU) data and the
  // corresponding data on the device (GPU).
  void **host_buffers_;
  void **device_buffers_;

  // Used to assign buffer IDs automatically by the system if the user
  // specifies an invalid buffer ID.
  int next_buffer_;

  cudaStream_t *streams_;
  int n_streams_;
  int last_stream_id_;

  int host_to_device_cb_idx_;
  int kernel_cb_idx_;
  int device_to_host_cb_idx_;
  int light_cb_idx_; // for lightweight version

  int running_kernel_idx_;
  int data_setup_idx_;
  int data_cleanup_idx_;

#ifdef HAPI_TRACE
  gpuEventTimer gpu_events_[QUEUE_SIZE_INIT * 3];
  std::atomic<int> time_idx_;
#endif

#ifdef HAPI_INSTRUMENT_WRS
  CkVec<CkVec<CkVec<hapiRequestTimeInfo> > > avg_times_;
  bool init_instr_;
#endif

#if CMK_SMP || CMK_MULTICORE
  CmiNodeLock buffer_lock_;
  CmiNodeLock queue_lock_;
  CmiNodeLock progress_lock_;
  CmiNodeLock stream_lock_;
#endif

  cudaDeviceProp device_prop_;
#ifdef HAPI_CUDA_CALLBACK
  bool cb_support;
#endif

  void init();
  int createStreams();
  void destroyStreams();
  cudaStream_t getNextStream();
  cudaStream_t getStream(int);
  void allocateBuffers(hapiWorkRequest*);
  void hostToDeviceTransfer(hapiWorkRequest*);
  void deviceToHostTransfer(hapiWorkRequest*);
  void freeBuffers(hapiWorkRequest*);
  void runKernel(hapiWorkRequest*);
};

// Declare GPU Manager as a process-shared object.
CsvDeclare(GPUManager, gpu_manager);

void GPUManager::init() {
  next_buffer_ = NUM_BUFFERS;
  streams_ = NULL;
  last_stream_id_ = -1;
  running_kernel_idx_ = 0;
  data_setup_idx_ = 0;
  data_cleanup_idx_ = 0;

#if CMK_SMP || CMK_MULTICORE
  // create mutex locks
  buffer_lock_ = CmiCreateLock();
  queue_lock_ = CmiCreateLock();
  progress_lock_ = CmiCreateLock();
  stream_lock_ = CmiCreateLock();
#endif

#ifdef HAPI_TRACE
  time_idx_ = 0;
#endif

  // store CUDA device properties
  hapiCheck(cudaGetDeviceProperties(&device_prop_, getMyCudaDevice(CmiMyPe())));

#ifdef HAPI_CUDA_CALLBACK
  // check if CUDA callback is supported
  // CUDA 5.0 (compute capability 3.0) or newer
  cb_support = (device_prop_.major >= 3);
  if (!cb_support) {
    CmiAbort("[HAPI] CUDA callback is not supported on this device");
  }
#endif

  // set which device to use
  hapiCheck(cudaSetDevice(getMyCudaDevice(CmiMyPe())));

  // allocate host/device buffers array (both user and system-addressed)
  host_buffers_ = new void*[NUM_BUFFERS*2];
  device_buffers_ = new void*[NUM_BUFFERS*2];

  // initialize device array to NULL
  for (int i = 0; i < NUM_BUFFERS*2; i++) {
    device_buffers_[i] = NULL;
  }

#ifdef HAPI_TRACE
  traceRegisterUserEvent("GPU Memory Setup", GpuMemSetup);
  traceRegisterUserEvent("GPU Kernel Execution", GpuKernelExec);
  traceRegisterUserEvent("GPU Memory Cleanup", GpuMemCleanup);
#endif

#ifdef HAPI_MEMPOOL
  mempool_boundaries_.reserve(HAPI_MEMPOOL_NUM_SLOTS);
  mempool_boundaries_.length() = HAPI_MEMPOOL_NUM_SLOTS;

  size_t buf_size = HAPI_MEMPOOL_MIN_BUFFER_SIZE;
  for(int i = 0; i < HAPI_MEMPOOL_NUM_SLOTS; i++){
    mempool_boundaries_[i] = buf_size;
    buf_size = buf_size << 1;
  }
#endif // HAPI_MEMPOOL

#ifdef HAPI_INSTRUMENT_WRS
  init_instr_ = false;
#endif
}

// Creates streams equal to the maximum number of concurrent kernels,
// which depends on the compute capability of the device.
// Returns the number of created streams.
int GPUManager::createStreams() {
  if (streams_)
    return n_streams_;

#if CMK_SMP || CMK_MULTICORE
  if (device_prop_.major == 3) {
    if (device_prop_.minor == 0)
      n_streams_ = 16;
    else if (device_prop_.minor == 2)
      n_streams_ = 4;
    else // 3.5, 3.7 or unknown 3.x
      n_streams_ = 32;
  }
  else if (device_prop_.major == 5) {
    if (device_prop_.minor == 3)
      n_streams_ = 16;
    else // 5.0, 5.2 or unknown 5.x
      n_streams_ = 32;
  }
  else if (device_prop_.major == 6) {
    if (device_prop_.minor == 1)
      n_streams_ = 32;
    else if (device_prop_.minor == 2)
      n_streams_ = 16;
    else // 6.0 or unknown 6.x
      n_streams_ = 128;
  }
  else // unknown (future) compute capability
    n_streams_ = 128;
#else
  n_streams_ = 4; // FIXME per PE in non-SMP mode
#endif

  streams_ = new cudaStream_t[n_streams_];
  for (int i = 0; i < n_streams_; i++) {
    hapiCheck(cudaStreamCreate(&streams_[i]));
  }

  return n_streams_;
}

void GPUManager::destroyStreams() {
  if (streams_) {
    for (int i = 0; i < n_streams_; i++) {
      hapiCheck(cudaStreamDestroy(streams_[i]));
    }
  }
}

cudaStream_t GPUManager::getNextStream() {
  if (streams_ == NULL)
    return NULL;

  last_stream_id_ = (++last_stream_id_) % n_streams_;
  return streams_[last_stream_id_];
}

cudaStream_t GPUManager::getStream(int i) {
  if (streams_ == NULL)
    return NULL;

  if (i < 0 || i >= n_streams_)
    CmiAbort("[HAPI] invalid stream ID");
  return streams_[i];
}

// Allocates device buffers.
void GPUManager::allocateBuffers(hapiWorkRequest* wr) {
  for (int i = 0; i < wr->getBufferCount(); i++) {
    hapiBufferInfo& bi = wr->buffers[i];
    int index = bi.id;
    int size = bi.size;

    // if index value is invalid, use an available ID
    if (index < 0 || index >= NUM_BUFFERS) {
      bool is_found = false;
      for (int j = next_buffer_; j < NUM_BUFFERS*2; j++) {
        if (device_buffers_[j] == NULL) {
          index = j;
          is_found = true;
          break;
        }
      }

      // if no index was found, try to search for a value at the
      // beginning of the system addressed space
      if (!is_found) {
        for (int j = NUM_BUFFERS; j < next_buffer_; j++) {
          if (device_buffers_[j] == NULL) {
            index = j;
            is_found = true;
            break;
          }
        }
      }

      if (!is_found) {
        CmiAbort("[HAPI] ran out of device buffer indices");
      }

      next_buffer_ = index + 1;
      if (next_buffer_ == NUM_BUFFERS*2) {
        next_buffer_ = NUM_BUFFERS;
      }

      bi.id = index;
    }

    if (device_buffers_[index] == NULL && size > 0) {
      // allocate device memory
      hapiCheck(cudaMalloc((void **)&device_buffers_[index], size));

#ifdef HAPI_DEBUG
      printf("[HAPI] allocated buffer %d at %p, time: %.2f, size: %d\n",
             index, device_buffers_[index], cutGetTimerValue(timerHandle),
             size);
#endif
    }
  }
}

#ifndef HAPI_CUDA_CALLBACK
void recordEvent(cudaStream_t stream, void* cb, void* cb_msg) {
  // create CUDA event and insert into stream
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaEventRecord(ev, stream);

  hapiEvent hev;
  hev.event = ev;
  hev.cb = cb;
  hev.cb_msg = cb_msg;

  // push event information in queue
  CpvAccess(hapi_event_queue).push(hev);

  // increase count so that scheduler can poll the queue
  CpvAccess(n_hapi_events)++;
}
#endif

// Initiates host-to-device data transfer.
void GPUManager::hostToDeviceTransfer(hapiWorkRequest* wr) {
  for (int i = 0; i < wr->getBufferCount(); i++) {
    hapiBufferInfo& bi = wr->buffers[i];
    int index = bi.id;
    int size = bi.size;
    host_buffers_[index] = bi.host_buffer;

    if (bi.transfer_to_device && size > 0) {
      hapiCheck(cudaMemcpy(device_buffers_[index], host_buffers_[index], size,
                                cudaMemcpyHostToDevice));

#ifdef HAPI_DEBUG
      printf("[HAPI] transferring buffer %d from host to device, time: %.2f, "
             "size: %d\n", index, cutGetTimerValue(timerHandle), size);
#endif
    }
  }
}

// Initiates device-to-host data transfer.
void GPUManager::deviceToHostTransfer(hapiWorkRequest* wr) {
  for (int i = 0; i < wr->getBufferCount(); i++) {
    hapiBufferInfo& bi = wr->buffers[i];
    int index = bi.id;
    int size = bi.size;

    if (bi.transfer_to_host && size > 0) {
      hapiCheck(cudaMemcpy(host_buffers_[index], device_buffers_[index], size,
                                cudaMemcpyDeviceToHost));

#ifdef HAPI_DEBUG
      printf("[HAPI] transferring buffer %d from device to host, time %.2f, "
             "size: %d\n", index, cutGetTimerValue(timerHandle), size);
#endif
    }
  }
}

// Frees device buffers.
void GPUManager::freeBuffers(hapiWorkRequest* wr) {
  for (int i = 0; i < wr->getBufferCount(); i++) {
    hapiBufferInfo& bi = wr->buffers[i];
    int index = bi.id;

    if (bi.need_free) {
      hapiCheck(cudaFree(device_buffers_[index]));
      device_buffers_[index] = NULL;

#ifdef HAPI_DEBUG
      printf("[HAPI] freed buffer %d, time %.2f\n",
             index, cutGetTimerValue(timerHandle));
#endif
    }
  }
}

inline void lockAndFreeBuffersDeleteWr(hapiWorkRequest* wr) {
#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // free device buffers
  CsvAccess(gpu_manager).freeBuffers(wr);

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // free hapiWorkRequest
  delete wr;
}

// Run the user's kernel for the given work request.
// This used to be a switch statement defined by the user to allow the runtime
// to execute the correct kernel.
void GPUManager::runKernel(hapiWorkRequest* wr) {
	if (wr->runKernel) {
		wr->runKernel(wr, wr->stream, device_buffers_);
	}
	// else, might be only for data transfer (or might be a bug?)
}

// Invokes user's host-to-device callback.
static void* hostToDeviceCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("hostToDeviceCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  CUDACallbackManager(wr->host_to_device_cb);

  return NULL;
}

// Invokes user's kernel execution callback.
static void* kernelCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("kernelCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  CUDACallbackManager(wr->kernel_cb);

  return NULL;
}

// Frees device buffers and invokes user's device-to-host callback.
// Invoked regardless of the availability of the user's callback.
static void* deviceToHostCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("deviceToHostCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));

  // invoke user callback
  if (wr->device_to_host_cb) {
    CUDACallbackManager(wr->device_to_host_cb);
  }

  lockAndFreeBuffersDeleteWr(wr);

  // notify process to QD
  QdProcess(1);

  return NULL;
}

// Used by lightweight HAPI.
static void* lightCallback(void *arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("lightCallback", NVTXColor::Asbestos);
#endif

  char* conv_msg_tmp = (char*)arg + CmiMsgHeaderSizeBytes + sizeof(int);
  void* cb = *((void**)conv_msg_tmp);

  // invoke user callback
  if (cb != NULL) {
    CUDACallbackManager(cb);
  }

  // notify process to QD
  QdProcess(1);

  return NULL;
}

// Register callback functions. All PEs need to call this.
void hapiRegisterCallbacks() {
  // FIXME: Potential race condition on assignments, but CmiAssignOnce
  // causes a hang at startup.
  CsvAccess(gpu_manager).host_to_device_cb_idx_
    = CmiRegisterHandler((CmiHandler)hostToDeviceCallback);
  CsvAccess(gpu_manager).kernel_cb_idx_
    = CmiRegisterHandler((CmiHandler)kernelCallback);
  CsvAccess(gpu_manager).device_to_host_cb_idx_
    = CmiRegisterHandler((CmiHandler)deviceToHostCallback);
  CsvAccess(gpu_manager).light_cb_idx_
    = CmiRegisterHandler((CmiHandler)lightCallback);
}

// Callback function invoked by the CUDA runtime certain parts of GPU work are
// complete. It sends a converse message to the original PE to free the relevant
// device memory and invoke the user's callback. The reason for this method is
// that a thread created by the CUDA runtime does not have access to any of the
// CpvDeclare'd variables as it is not one of the threads created by the Charm++
// runtime.
static void CUDART_CB CUDACallback(cudaStream_t stream, cudaError_t status,
                                   void *data) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("CUDACallback", NVTXColor::Silver);
#endif

  if (status == cudaSuccess) {
    // send message to the original PE
    char *conv_msg = (char*)data;
    int dstRank = *((int *)(conv_msg + CmiMsgHeaderSizeBytes));
    CmiPushPE(dstRank, conv_msg);
  }
  else {
    CmiAbort("[HAPI] error before CUDACallback");
  }
}

#ifdef HAPI_CUDA_CALLBACK
enum CallbackStage {
  AfterHostToDevice,
  AfterKernel,
  AfterDeviceToHost
};

static void addCallback(hapiWorkRequest *wr, CallbackStage stage) {
  // create converse message to be delivered to this PE after CUDA callback
  char *conv_msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes + sizeof(int) +
                                  sizeof(hapiWorkRequest *)); // FIXME memory leak?
  *((int *)(conv_msg + CmiMsgHeaderSizeBytes)) = CmiMyRank();
  *((hapiWorkRequest **)(conv_msg + CmiMsgHeaderSizeBytes + sizeof(int))) = wr;

  int handlerIdx;
  switch (stage) {
    case AfterHostToDevice:
      handlerIdx = CsvAccess(gpu_manager).host_to_device_cb_idx_;
      break;
    case AfterKernel:
      handlerIdx = CsvAccess(gpu_manager).kernel_cb_idx_;
      break;
    case AfterDeviceToHost:
      handlerIdx = CsvAccess(gpu_manager).device_to_host_cb_idx_;
      break;
    default: // wrong type
      CmiFree(conv_msg);
      return;
  }
  CmiSetHandler(conv_msg, handlerIdx);

  // add callback into CUDA stream
  hapiCheck(cudaStreamAddCallback(wr->stream, CUDACallback, (void*)conv_msg, 0));
}
#endif // HAPI_CUDA_CALLBACK

/******************** DEPRECATED ********************/
// User calls this function to offload work to the GPU.
void hapiEnqueue(hapiWorkRequest* wr) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("enqueue", NVTXColor::Pomegranate);
#endif

  // notify create to QD
  QdCreate(1);

#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // allocate device memory
  CsvAccess(gpu_manager).allocateBuffers(wr);

  // transfer data to device
  CsvAccess(gpu_manager).hostToDeviceTransfer(wr);

  // add host-to-device transfer callback
  if (wr->host_to_device_cb) {
#ifdef HAPI_CUDA_CALLBACK
    addCallback(wr, AfterHostToDevice);
#else
    recordEvent(wr->stream, wr->host_to_device_cb, NULL);
#endif
  }

  // run kernel
  CsvAccess(gpu_manager).runKernel(wr);

  // add kernel callback
  if (wr->kernel_cb) {
#ifdef HAPI_CUDA_CALLBACK
    addCallback(wr, AfterKernel);
#else
    recordEvent(wr->stream, wr->kernel_cb, NULL);
#endif
  }

  // transfer data to host
  CsvAccess(gpu_manager).deviceToHostTransfer(wr);

  // add device-to-host transfer callback
#ifdef HAPI_CUDA_CALLBACK
  // always invoked to free memory
  addCallback(wr, AfterDeviceToHost);
#else
  if (wr->device_to_host_cb) {
    recordEvent(wr->stream, wr->device_to_host_cb, NULL);
  }

  // free device buffers
  recordEvent(wr->stream, FREE_BUFF_FLAG, wr);
#endif

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).progress_lock_);
#endif
}

/******************** DEPRECATED ********************/
// Creates a hapiWorkRequest object on the heap and returns it to the user.
hapiWorkRequest* hapiCreateWorkRequest() {
  return (new hapiWorkRequest);
}

#ifdef HAPI_MEMPOOL
static void createPool(int *nbuffers, int n_slots, CkVec<BufferPool> &pools);
static void releasePool(CkVec<BufferPool> &pools);
#endif

// Initialization of HAPI functionalities.
void initHybridAPI() {
  // create and initialize GPU Manager object
  CsvInitialize(GPUManager, gpu_manager);
  CsvAccess(gpu_manager).init();

#ifdef HAPI_MEMPOOL
  // create pool of page-locked memory
  int sizes[HAPI_MEMPOOL_NUM_SLOTS];
        /*256*/ sizes[0]  =  4;
        /*512*/ sizes[1]  =  2;
       /*1024*/ sizes[2]  =  2;
       /*2048*/ sizes[3]  =  4;
       /*4096*/ sizes[4]  =  2;
       /*8192*/ sizes[5]  =  6;
      /*16384*/ sizes[6]  =  5;
      /*32768*/ sizes[7]  =  2;
      /*65536*/ sizes[8]  =  1;
     /*131072*/ sizes[9]  =  1;
     /*262144*/ sizes[10] =  1;
     /*524288*/ sizes[11] =  1;
    /*1048576*/ sizes[12] =  1;
    /*2097152*/ sizes[13] =  2;
    /*4194304*/ sizes[14] =  2;
    /*8388608*/ sizes[15] =  2;
   /*16777216*/ sizes[16] =  2;
   /*33554432*/ sizes[17] =  1;
   /*67108864*/ sizes[18] =  1;
  /*134217728*/ sizes[19] =  7;
  createPool(sizes, HAPI_MEMPOOL_NUM_SLOTS, CsvAccess(gpu_manager).mempool_free_bufs_);

#ifdef HAPI_MEMPOOL_DEBUG
  printf("[HAPI (%d)] done creating buffer pool\n", CmiMyPe());
#endif

#endif // HAPI_MEMPOOL
}

// Clean up and delete memory used by HAPI.
void exitHybridAPI() {
#if CMK_SMP || CMK_MULTICORE
  // destroy mutex locks
  CmiDestroyLock(CsvAccess(gpu_manager).buffer_lock_);
  CmiDestroyLock(CsvAccess(gpu_manager).queue_lock_);
  CmiDestroyLock(CsvAccess(gpu_manager).progress_lock_);
  CmiDestroyLock(CsvAccess(gpu_manager).stream_lock_);
#endif

  // destroy streams (if they were created)
  CsvAccess(gpu_manager).destroyStreams();

#ifdef HAPI_MEMPOOL
  // release memory pool
  releasePool(CsvAccess(gpu_manager).mempool_free_bufs_);
#endif // HAPI_MEMPOOL

#ifdef HAPI_TRACE
  for (int i = 0; i < CsvAccess(gpu_manager).time_idx_; i++) {
    switch (CsvAccess(gpu_manager).gpu_events_[i].event_type) {
    case DataSetup:
      printf("[HAPI] kernel %s data setup\n",
             CsvAccess(gpu_manager).gpu_events_[i].trace_name);
      break;
    case DataCleanup:
      printf("[HAPI] kernel %s data cleanup\n",
             CsvAccess(gpu_manager).gpu_events_[i].trace_name);
      break;
    case KernelExecution:
      printf("[HAPI] kernel %s execution\n",
             CsvAccess(gpu_manager).gpu_events_[i].trace_name);
      break;
    default:
      printf("[HAPI] invalid timer identifier\n");
    }
    printf("[HAPI] %.2f:%.2f\n",
           CsvAccess(gpu_manager).gpu_events_[i].cmi_start_time -
           CsvAccess(gpu_manager).gpu_events_[0].cmi_start_time,
           CsvAccess(gpu_manager).gpu_events_[i].cmi_end_time -
           CsvAccess(gpu_manager).gpu_events_[0].cmi_start_time);
  }
#endif
}

/******************** DEPRECATED ********************/
// Need to be updated with the Tracing API.
static inline void gpuEventStart(hapiWorkRequest* wr, int* index,
                                 WorkRequestStage event, ProfilingStage stage) {
#ifdef HAPI_TRACE
  gpuEventTimer* shared_gpu_events_ = CsvAccess(gpu_manager).gpu_events_;
  int shared_time_idx_ = CsvAccess(gpu_manager).time_idx_++;
  shared_gpu_events_[shared_time_idx_].cmi_start_time = CmiWallTimer();
  shared_gpu_events_[shared_time_idx_].event_type = event;
  shared_gpu_events_[shared_time_idx_].trace_name = wr->trace_name;
  *index = shared_time_idx_;
  shared_gpu_events_[shared_time_idx_].stage = stage;
#ifdef HAPI_DEBUG
  printf("[HAPI] start event %d of WR %s, profiling stage %d\n",
         event, wr->trace_name, stage);
#endif
#endif // HAPI_TRACE
}

/******************** DEPRECATED ********************/
// Need to be updated with the Tracing API.
static inline void gpuEventEnd(int index) {
#ifdef HAPI_TRACE
  CsvAccess(gpu_manager).gpu_events_[index].cmi_end_time = CmiWallTimer();
  traceUserBracketEvent(CsvAccess(gpu_manager).gpu_events_[index].stage,
                        CsvAccess(gpu_manager).gpu_events_[index].cmi_start_time,
                        CsvAccess(gpu_manager).gpu_events_[index].cmi_end_time);
#ifdef HAPI_DEBUG
  printf("[HAPI] end event %d of WR %s, profiling stage %d\n",
          CsvAccess(gpu_manager).gpu_events_[index].event_type,
          CsvAccess(gpu_manager).gpu_events_[index].trace_name,
          CsvAccess(gpu_manager).gpu_events_[index].stage);
#endif
#endif // HAPI_TRACE
}

static inline void hapiWorkRequestStartTime(hapiWorkRequest* wr) {
#ifdef HAPI_INSTRUMENT_WRS
  wr->phase_start_time = CmiWallTimer();
#endif
}

static inline void profileWorkRequestEvent(hapiWorkRequest* wr,
                                           WorkRequestStage event) {
#ifdef HAPI_INSTRUMENT_WRS
  if (initializedInstrument()) {
    double tt = CmiWallTimer() - (wr->phase_start_time);
    int index = wr->chare_index;
    char type = wr->comp_type;
    char phase = wr->comp_phase;

    CkVec<hapiRequestTimeInfo> &vec = wr->avg_times_[index][type];
    if (vec.length() <= phase){
      vec.growAtLeast(phase);
      vec.length() = phase+1;
    }
    switch (event) {
      case DataSetup:
        vec[phase].transfer_time += tt;
        break;
      case KernelExecution:
        vec[phase].kernel_time += tt;
        break;
      case DataCleanup:
        vec[phase].cleanup_time += tt;
        vec[phase].n++;
        break;
      default:
        printf("[HAPI] invalid event during profileWorkRequestEvent\n");
    }
  }
#endif
}

#ifdef HAPI_MEMPOOL
// Create a pool with n_slots slots.
// There are n_buffers[i] buffers for each buffer size corresponding to entry i.
// TODO list the alignment/fragmentation issues with either of two allocation schemes:
// if single, large buffer is allocated for each subpool
// if multiple, smaller buffers are allocated for each subpool
static void createPool(int *n_buffers, int n_slots, CkVec<BufferPool> &pools){
  CkVec<size_t>& mempool_boundaries = CsvAccess(gpu_manager).mempool_boundaries_;

  // initialize pools
  pools.reserve(n_slots);
  pools.length() = n_slots;
  for (int i = 0; i < n_slots; i++) {
    pools[i].size = mempool_boundaries[i];
    pools[i].head = NULL;
  }

  // divide by # of PEs on physical node and multiply by # of PEs in logical node
  size_t available_memory = CsvAccess(gpu_manager).device_prop_.totalGlobalMem /
                           CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyPe()))
                           * CmiMyNodeSize() * HAPI_MEMPOOL_SCALE;

  // pre-calculate memory per size
  int max_buffers = *std::max_element(n_buffers, n_buffers + n_slots);
  int n_buffers_to_allocate[n_slots];
  memset(n_buffers_to_allocate, 0, sizeof(n_buffers_to_allocate));
  size_t buf_size;
  while (available_memory >= mempool_boundaries[0] + sizeof(Header)) {
    for (int i = 0; i < max_buffers; i++) {
      for (int j = n_slots - 1; j >= 0; j--) {
        buf_size = mempool_boundaries[j] + sizeof(Header);
        if (i < n_buffers[j] && buf_size <= available_memory) {
          n_buffers_to_allocate[j]++;
          available_memory -= buf_size;
        }
      }
    }
  }

  // pin the host memory
  for (int i = 0; i < n_slots; i++) {
    buf_size = mempool_boundaries[i] + sizeof(Header);
    int num_buffers = n_buffers_to_allocate[i];

    Header* hd;
    Header* previous = NULL;

    // pin host memory in a contiguous block for a slot
    void* pinned_chunk;
    hapiCheck(cudaMallocHost(&pinned_chunk, buf_size * num_buffers));

    // initialize header structs
    for (int j = num_buffers - 1; j >= 0; j--) {
      hd = reinterpret_cast<Header*>(reinterpret_cast<unsigned char*>(pinned_chunk)
                                     + buf_size * j);
      hd->slot = i;
      hd->next = previous;
      previous = hd;
    }

    pools[i].head = previous;
#ifdef HAPI_MEMPOOL_DEBUG
    pools[i].num = num_buffers;
#endif
  }
}

static void releasePool(CkVec<BufferPool> &pools){
  for (int i = 0; i < pools.length(); i++) {
    Header* hdr = pools[i].head;
    if (hdr != NULL) {
      hapiCheck(cudaFreeHost((void*)hdr));
    }
  }
  pools.free();
}

static int findPool(int size){
  int boundary_array_len = CsvAccess(gpu_manager).mempool_boundaries_.length();
  if (size <= CsvAccess(gpu_manager).mempool_boundaries_[0]) {
    return 0;
  }
  else if (size > CsvAccess(gpu_manager).mempool_boundaries_[boundary_array_len-1]) {
    // create new slot
    CsvAccess(gpu_manager).mempool_boundaries_.push_back(size);

    BufferPool newpool;
    hapiCheck(cudaMallocHost((void**)&newpool.head, size + sizeof(Header)));
    if (newpool.head == NULL) {
      printf("[HAPI (%d)] findPool: failed to allocate newpool %d head, size %d\n",
             CmiMyPe(), boundary_array_len, size);
      CmiAbort("[HAPI] failed newpool allocation");
    }
    newpool.size = size;
#ifdef HAPI_MEMPOOL_DEBUG
    newpool.num = 1;
#endif
    CsvAccess(gpu_manager).mempool_free_bufs_.push_back(newpool);

    Header* hd = newpool.head;
    hd->next = NULL;
    hd->slot = boundary_array_len;

    return boundary_array_len;
  }
  for (int i = 0; i < CsvAccess(gpu_manager).mempool_boundaries_.length()-1; i++) {
    if (CsvAccess(gpu_manager).mempool_boundaries_[i] < size &&
        size <= CsvAccess(gpu_manager).mempool_boundaries_[i+1]) {
      return (i + 1);
    }
  }
  return -1;
}

static void* getBufferFromPool(int pool, int size){
  Header* ret;

  if (pool < 0 || pool >= CsvAccess(gpu_manager).mempool_free_bufs_.length()) {
    printf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %d invalid pool\n",
           CmiMyPe(), pool, size);
#ifdef HAPI_MEMPOOL_DEBUG
    printf("[HAPI (%d)] num: %d\n", CmiMyPe(),
           CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif
    CmiAbort("[HAPI] exiting after invalid pool");
  }
  else if (CsvAccess(gpu_manager).mempool_free_bufs_[pool].head == NULL) {
    Header* hd;
    hapiCheck(cudaMallocHost((void**)&hd, sizeof(Header) +
                             CsvAccess(gpu_manager).mempool_free_bufs_[pool].size));
#ifdef HAPI_MEMPOOL_DEBUG
    printf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %d expand by 1\n",
           CmiMyPe(), pool, size);
#endif
    if (hd == NULL) {
      CmiAbort("[HAPI] exiting after NULL hd from pool");
    }
    hd->slot = pool;
    return (void*)(hd + 1);
  }
  else {
    ret = CsvAccess(gpu_manager).mempool_free_bufs_[pool].head;
    CsvAccess(gpu_manager).mempool_free_bufs_[pool].head = ret->next;
#ifdef HAPI_MEMPOOL_DEBUG
    ret->size = size;
    CsvAccess(gpu_manager).mempool_free_bufs_[pool].num--;
#endif
    return (void*)(ret + 1);
  }
  return NULL;
}

static void returnBufferToPool(int pool, Header* hd) {
  hd->next = CsvAccess(gpu_manager).mempool_free_bufs_[pool].head;
  CsvAccess(gpu_manager).mempool_free_bufs_[pool].head = hd;
#ifdef HAPI_MEMPOOL_DEBUG
  CsvAccess(gpu_manager).mempool_free_bufs_[pool].num++;
#endif
}

void* hapiPoolMalloc(int size) {
#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).buffer_lock_);
#endif

  int pool = findPool(size);
  void* buf = getBufferFromPool(pool, size);

#ifdef HAPI_MEMPOOL_DEBUG
  printf("[HAPI (%d)] hapiPoolMalloc size %d pool %d left %d\n",
         CmiMyPe(), size, pool,
         CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).buffer_lock_);
#endif

  return buf;
}

void hapiPoolFree(void* ptr) {
  Header* hd = ((Header*)ptr) - 1;
  int pool = hd->slot;

#ifdef HAPI_MEMPOOL_DEBUG
  int size = hd->size;
#endif

#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).buffer_lock_);
#endif

  returnBufferToPool(pool, hd);

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).buffer_lock_);
#endif

#ifdef HAPI_MEMPOOL_DEBUG
  printf("[HAPI (%d)] hapiPoolFree size %d pool %d left %d\n",
         CmiMyPe(), size, pool,
         CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif
}
#endif // HAPI_MEMPOOL

#ifdef HAPI_INSTRUMENT_WRS
void hapiInitInstrument(int n_chares, char n_types) {
  avg_times_.reserve(n_chares);
  avg_times_.length() = n_chares;
  for (int i = 0; i < n_chares; i++) {
    avg_times_[i].reserve(n_types);
    avg_times_[i].length() = n_types;
  }
  init_instr_ = true;
}

static bool initializedInstrument() {
  return init_instr_;
}

hapiRequestTimeInfo* hapiQueryInstrument(int chare, char type, char phase) {
  if (phase < avg_times_[chare][type].length()) {
    return &avg_times_[chare][type][phase];
  }
  else {
    return NULL;
  }
}

void hapiClearInstrument() {
  for (int chare = 0; chare < avg_times_.length(); chare++) {
    for (int type = 0; type < avg_times_[chare].length(); type++) {
      for (int phase = 0; phase < avg_times_[chare][type].length(); phase++) {
        avg_times_[chare][type][phase].transferTime = 0.0;
        avg_times_[chare][type][phase].kernelTime = 0.0;
        avg_times_[chare][type][phase].cleanupTime = 0.0;
        avg_times_[chare][type][phase].n = 0;
      }
      avg_times_[chare][type].length() = 0;
    }
    avg_times_[chare].length() = 0;
  }
  avg_times_.length() = 0;
  init_instr_ = false;
}
#endif // HAPI_INSTRUMENT_WRS

void hapiPollEvents() {
#ifndef HAPI_CUDA_CALLBACK
  std::queue<hapiEvent>& queue = CpvAccess(hapi_event_queue);
  while (!queue.empty()) {
    hapiEvent hev = queue.front();
    if (cudaEventQuery(hev.event) == cudaSuccess) {
      // Check that this event isn't a special case i.e. just doing a callback
      if (hev.cb != FREE_BUFF_FLAG) {
        ((CkCallback*)hev.cb)->send(hev.cb_msg);
      }
      // Hack cb field to serve as a flag and cb_msg as a pointer to the hapiWorkRequest
      else { // free buffers
        lockAndFreeBuffersDeleteWr(static_cast<hapiWorkRequest*>(hev.cb_msg));
      }
      cudaEventDestroy(hev.event);
      queue.pop();
      CpvAccess(n_hapi_events)--;
    }
    else {
      // FIXME maybe we should make one pass of all entries?
      break;
    }
  }
#endif
}

int hapiCreateStreams() {
#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).stream_lock_);
#endif

  int ret = CsvAccess(gpu_manager).createStreams();

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).stream_lock_);
#endif

  return ret;
}

cudaStream_t hapiGetStream() {
#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).stream_lock_);
#endif

  cudaStream_t ret = CsvAccess(gpu_manager).getNextStream();

#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).stream_lock_);
#endif

  return ret;
}

// Lightweight HAPI, to be invoked after data transfer or kernel execution.
void hapiAddCallback(cudaStream_t stream, void* cb, void* cb_msg) {
#ifndef HAPI_CUDA_CALLBACK
  // record CUDA event
  recordEvent(stream, cb, cb_msg);
#else
  /* FIXME works for now (faster too), but CmiAlloc might not be thread-safe
#if CMK_SMP || CMK_MULTICORE
  CmiLock(CsvAccess(gpu_manager).queue_lock_);
#endif
*/

  // create converse message to be delivered to this PE after CUDA callback
  char* conv_msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes + sizeof(int) +
                                 sizeof(void*)); // FIXME memory leak?
  char* conv_msg_tmp = conv_msg + CmiMsgHeaderSizeBytes;
  *((int*)conv_msg_tmp) = CmiMyRank();
  conv_msg_tmp += sizeof(int);
  *((void**)conv_msg_tmp) = cb;
  CmiSetHandler(conv_msg, CsvAccess(gpu_manager).light_cb_idx_);

  // push into CUDA stream
  hapiCheck(cudaStreamAddCallback(stream, CUDACallback, (void*)conv_msg, 0));

  /*
#if CMK_SMP || CMK_MULTICORE
  CmiUnlock(CsvAccess(gpu_manager).queue_lock_);
#endif
*/
#endif

  // notify create to QD
  QdCreate(1);
}

cudaError_t hapiMalloc(void** devPtr, size_t size) {
  return cudaMalloc(devPtr, size);
}

cudaError_t hapiFree(void* devPtr) {
  return cudaFree(devPtr);
}

cudaError_t hapiMallocHost(void** ptr, size_t size) {
  return cudaMallocHost(ptr, size);
}

cudaError_t hapiFreeHost(void* ptr) {
  return cudaFreeHost(ptr);
}

cudaError_t hapiMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

void hapiErrorDie(cudaError_t retCode, const char* code, const char* file, int line) {
  if (retCode != cudaSuccess) {
    fprintf(stderr, "Fatal CUDA Error [%d] %s at %s:%d\n", retCode, cudaGetErrorString(retCode), file, line);
    CmiAbort("Exit due to CUDA error");
  }
}
