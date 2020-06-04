#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <atomic>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda_runtime.h>

#include "converse.h"
#include "hapi.h"
#include "hapi_impl.h"
#include "gpumanager.h"
#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
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

static void createPool(int *nbuffers, int n_slots, std::vector<BufferPool> &pools);
static void releasePool(std::vector<BufferPool> &pools);

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

#ifdef HAPI_CUDA_CALLBACK
struct hapiCallbackMessage {
  char header[CmiMsgHeaderSizeBytes];
  int rank;
  void* cb;
  void* cb_msg;
};
#endif

#ifndef HAPI_CUDA_CALLBACK
typedef struct hapiEvent {
  cudaEvent_t event;
  void* cb;
  void* cb_msg;
  hapiWorkRequest* wr; // if this is not NULL, buffers and request itself are deallocated

  hapiEvent(cudaEvent_t event_, void* cb_, void* cb_msg_, hapiWorkRequest* wr_ = NULL)
            : event(event_), cb(cb_), cb_msg(cb_msg_), wr(wr_) {}
} hapiEvent;

CpvDeclare(std::queue<hapiEvent>, hapi_event_queue);
#endif // HAPI_CUDA_CALLBACK
CpvDeclare(int, n_hapi_events);

// Used to invoke user's Charm++ callback function
void (*hapiInvokeCallback)(void*, void*) = NULL;

// Functions used to support quiescence detection.
void (*hapiQdCreate)(int) = NULL;
void (*hapiQdProcess)(int) = NULL;

#define MAX_PINNED_REQ 64
#define MAX_DELAYED_FREE_REQS 64

// Declare GPU Manager as a process-shared object.
CsvDeclare(GPUManager, gpu_manager);

CpvDeclare(int, my_device); // GPU device that this thread is mapped to
CpvDeclare(bool, device_rep); // Is this PE a device representative thread? (1 per device)

// Returns the local rank of the logical node (process) that the given PE belongs to
static inline int CmiNodeRankLocal(int pe) {
  // Logical node index % Number of logical nodes per physical node
  return CmiNodeOf(pe) % (CmiNumNodes() / CmiNumPhysicalNodes());
}

// Returns the local rank of the logical node that I belong to
static inline int CmiMyNodeRankLocal() {
  return CmiNodeRankLocal(CmiMyPe());
}

// Initialize per-process variables
void hapiInitCsv() {
  // Create and initialize GPU Manager object
  CsvInitialize(GPUManager, gpu_manager);
  CsvAccess(gpu_manager).init();
}

// Initialize per-PE variables
void hapiInitCpv() {
  // HAPI event-related
#ifndef HAPI_CUDA_CALLBACK
  CpvInitialize(std::queue<hapiEvent>, hapi_event_queue);
#endif
  CpvInitialize(int, n_hapi_events);
  CpvAccess(n_hapi_events) = 0;

  // Device mapping
  CpvInitialize(int, my_device);
  CpvAccess(my_device) = 0;
  CpvInitialize(bool, device_rep);
  CpvAccess(device_rep) = false;
}

// Clean up per-process data
void hapiExitCsv() {
  // Destroy GPU Manager object
  CsvAccess(gpu_manager).destroy();

  // Release memory pool
  if (CsvAccess(gpu_manager).mempool_initialized_) {
    releasePool(CsvAccess(gpu_manager).mempool_free_bufs_);
  }
}

// Set up PE to GPU mapping, invoked from all PEs
// TODO: Support custom mappings
void hapiMapping(char** argv) {
  Mapping map_type = Mapping::Block; // Default is block mapping
  bool all_gpus = false; // If true, all GPUs are visible to all processes.
                         // Otherwise, only a subset are visible (e.g. with jsrun)
  char* gpumap = NULL;

  // Process +gpumap
  if (CmiGetArgStringDesc(argv, "+gpumap", &gpumap,
        "define pe to gpu device mapping")) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> PE-GPU mapping: %s\n", gpumap);
    }

    if (strcmp(gpumap, "none") == 0) {
      map_type = Mapping::None;
    } else if (strcmp(gpumap, "block") == 0) {
      map_type = Mapping::Block;
    } else if (strcmp(gpumap, "roundrobin") == 0) {
      map_type = Mapping::RoundRobin;
    } else {
      CmiAbort("Unsupported mapping type: %s, use one of \"none\", \"block\", "
          "\"roundrobin\"", gpumap);
    }
  }

  // Process +allgpus
  if (CmiGetArgFlagDesc(argv, "+allgpus",
        "all GPUs are visible to all processes")) {
    all_gpus = true;
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> All GPUs are visible to all processes\n");
    }
  }

  // No mapping specified, user assumes responsibility
  if (map_type == Mapping::None) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> User should explicitly select devices for PEs/chares\n");
    }
    return;
  }

  CmiAssert(map_type != Mapping::None);

  if (CmiMyRank() == 0) {
    // Count number of GPU devices used by each process
    int visible_device_count;
    hapiCheck(cudaGetDeviceCount(&visible_device_count));
    if (visible_device_count <= 0) {
      CmiAbort("Unable to perform PE-GPU mapping, no GPUs found!");
    }

    int& device_count = CsvAccess(gpu_manager).device_count;
    if (all_gpus) {
      device_count = visible_device_count / (CmiNumNodes() / CmiNumPhysicalNodes());
    } else {
      device_count = visible_device_count;
    }

    // Handle the case where the number of GPUs per process are larger than
    // the number of PEs per process. This is needed because we currently don't
    // support each PE using more than one device.
    if (device_count > CmiNodeSize(CmiMyNode())) {
      if (CmiMyPe() == 0) {
        CmiPrintf("HAPI> Found more GPU devices (%d) than PEs (%d) per process, "
            "limiting to %d device(s) per process\n", device_count,
            CmiNodeSize(CmiMyNode()), CmiNodeSize(CmiMyNode()));
      }
      device_count = CmiNodeSize(CmiMyNode());
    }

    // Create a DeviceManager per GPU device
    std::vector<DeviceManager>& device_managers = CsvAccess(gpu_manager).device_managers;
    for (int i = 0; i < device_count; i++) {
      device_managers.emplace_back(i, device_count * CmiMyNodeRankLocal() + i);
    }

    // Count number of PEs per device
    CsvAccess(gpu_manager).pes_per_device = CmiNodeSize(CmiMyNode()) / device_count;

    // Count number of devices on a physical node
    CsvAccess(gpu_manager).device_count_on_physical_node =
      device_count * (CmiNumNodes() / CmiNumPhysicalNodes());
  }

  if (CmiMyPe() == 0) {
    CmiPrintf("HAPI> Config: %d device(s) per process, %d PE(s) per device, %d device(s) per host\n",
        CsvAccess(gpu_manager).device_count, CsvAccess(gpu_manager).pes_per_device,
        CsvAccess(gpu_manager).device_count_on_physical_node);
  }

  CmiNodeBarrier();

  // Perform mapping and set device representative PE
  int my_rank = all_gpus ? CmiPhysicalRank(CmiMyPe()) : CmiMyRank();

  switch (map_type) {
    case Mapping::Block:
      CpvAccess(my_device) = my_rank / CsvAccess(gpu_manager).pes_per_device;
      if (my_rank % CsvAccess(gpu_manager).pes_per_device == 0) CpvAccess(device_rep) = true;
      break;
    case Mapping::RoundRobin:
      CpvAccess(my_device) = my_rank % CsvAccess(gpu_manager).device_count;
      if (my_rank < CsvAccess(gpu_manager).device_count) CpvAccess(device_rep) = true;
      break;
    default:
      CmiAbort("Unsupported mapping type!");
  }

  // Set device and store PE-device mapping
  hapiCheck(cudaSetDevice(CpvAccess(my_device)));
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).device_mapping_lock);
#endif
  CsvAccess(gpu_manager).device_map.emplace(CmiMyPe(),
      &(CsvAccess(gpu_manager).device_managers[CpvAccess(my_device)]));
#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).device_mapping_lock);
#endif

  // Process device communication buffer parameters (in MB)
  int input_comm_buffer_size;
  if (CmiGetArgIntDesc(argv, "+gpucommbuffer", &input_comm_buffer_size,
        "GPU communication buffer size (in MB)")) {
    if (CmiMyRank() == 0) {
      // Round up size to the closest power of 2
      size_t comm_buffer_size = (size_t)input_comm_buffer_size * 1024 * 1024;
      int size_log2 = std::ceil(std::log2((double)comm_buffer_size));
      CsvAccess(gpu_manager).comm_buffer_size = (size_t)std::pow(2, size_log2);
    }
  }

  if (CmiMyPe() == 0) {
    CmiPrintf("HAPI> GPU communication buffer size: %lu MB "
        "(rounded up to the nearest power of two)\n",
        CsvAccess(gpu_manager).comm_buffer_size / (1024 * 1024));
  }

  CmiNodeBarrier();

  // Create device communication buffers
  // Should only be done by device representative threads
  if (CpvAccess(device_rep)) {
    DeviceManager* dm = CsvAccess(gpu_manager).device_map[CmiMyPe()];
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    dm->create_comm_buffer(CsvAccess(gpu_manager).comm_buffer_size);
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
  }

  // Process custom size for CUDA IPC event pool
  int input_ipc_event_pool_size;
  if (!CmiGetArgIntDesc(argv, "+gpuipceventpool", &input_ipc_event_pool_size,
        "GPU IPC event pool size per PE")) {
    input_ipc_event_pool_size = 16;
  }

  if (CmiMyRank() == 0) {
    CsvAccess(gpu_manager).ipc_event_pool_size = input_ipc_event_pool_size * CsvAccess(gpu_manager).pes_per_device;
  }

  if (CmiMyPe() == 0) {
    CmiPrintf("HAPI> CUDA IPC event pool size - %d per PE, %d per device\n",
        input_ipc_event_pool_size, CsvAccess(gpu_manager).ipc_event_pool_size);
  }

  // Check if P2P access should be enabled
  bool enable_peer = true; // Enabled by default
  if (CmiGetArgFlagDesc(argv, "+nogpupeer",
        "do not enable P2P access between visible GPU pairs")) {
    enable_peer = false;
  }

  // Enable P2P access to other visible devices
  // (only useful for multiple devices per process)
  // Should only be done by device representative threads
  if (enable_peer) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> Enabling P2P access between devices\n");
    }
    if (CpvAccess(device_rep)) {
      for (int i = 0; i < CsvAccess(gpu_manager).device_count; i++) {
        if (i != CpvAccess(my_device)) {
          int can_access_peer;

          hapiCheck(cudaDeviceCanAccessPeer(&can_access_peer, CpvAccess(my_device), i));
          if (can_access_peer) {
            cudaDeviceEnablePeerAccess(i, 0);
          }
        }
      }
    }
  } else {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> P2P access between devices not enabled\n");
    }
  }
}

#ifndef HAPI_CUDA_CALLBACK
void recordEvent(cudaStream_t stream, void* cb, void* cb_msg, hapiWorkRequest* wr = NULL) {
  // create CUDA event and insert into stream
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaEventRecord(ev, stream);

  hapiEvent hev(ev, cb, cb_msg, wr);

  // push event information in queue
  CpvAccess(hapi_event_queue).push(hev);

  // increase count so that scheduler can poll the queue
  CpvAccess(n_hapi_events)++;
}
#endif

inline static void hapiWorkRequestCleanup(hapiWorkRequest* wr) {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // free device buffers
  CsvAccess(gpu_manager).freeBuffers(wr);

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // free hapiWorkRequest
  delete wr;
}

#ifdef HAPI_CUDA_CALLBACK
// Invokes user's host-to-device callback.
static void* hostToDeviceCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("hostToDeviceCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  CmiAssert(hapiInvokeCallback);
  hapiInvokeCallback(wr->host_to_device_cb, NULL);

  // inform QD that the host-to-device transfer is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}

// Invokes user's kernel execution callback.
static void* kernelCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("kernelCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  CmiAssert(hapiInvokeCallback);
  hapiInvokeCallback(wr->kernel_cb, NULL);

  // inform QD that the kernel is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

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
    CmiAssert(hapiInvokeCallback);
    hapiInvokeCallback(wr->device_to_host_cb, NULL);
  }

  hapiWorkRequestCleanup(wr);

  // inform QD that device-to-host transfer is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}

// Used by lightweight HAPI.
static void* lightCallback(void *arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("lightCallback", NVTXColor::Asbestos);
#endif

  hapiCallbackMessage* conv_msg = (hapiCallbackMessage*)arg;

  // invoke user callback
  if (conv_msg->cb) {
    CmiAssert(hapiInvokeCallback);
    hapiInvokeCallback(conv_msg->cb, conv_msg->cb_msg);
  }

  // notify process to QD
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}
#endif // HAPI_CUDA_CALLBACK

// Register callback functions. All PEs need to call this.
void hapiRegisterCallbacks() {
#ifdef HAPI_CUDA_CALLBACK
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
#endif
}

#ifdef HAPI_CUDA_CALLBACK
// Callback function invoked by the CUDA runtime certain parts of GPU work are
// complete. It sends a converse message to the original PE to free the relevant
// device memory and invoke the user's callback. The reason for this method is
// that a thread created by the CUDA runtime does not have access to any of the
// CpvDeclare'd variables as it is not one of the threads created by the Charm++
// runtime.
static void CUDACallback(void *data) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("CUDACallback", NVTXColor::Silver);
#endif

  // send message to the original PE
  char *conv_msg = (char*)data;
  int dstRank = *((int *)(conv_msg + CmiMsgHeaderSizeBytes));
  CmiPushPE(dstRank, conv_msg);
}

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
  hapiCheck(cudaLaunchHostFunc(wr->stream, CUDACallback, (void*)conv_msg));
}
#endif // HAPI_CUDA_CALLBACK

/******************** DEPRECATED ********************/
// User calls this function to offload work to the GPU.
void hapiEnqueue(hapiWorkRequest* wr) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("enqueue", NVTXColor::Pomegranate);
#endif

#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).progress_lock_);
#endif

  // allocate device memory
  CsvAccess(gpu_manager).allocateBuffers(wr);

  // transfer data to device
  CsvAccess(gpu_manager).hostToDeviceTransfer(wr);

  // add host-to-device transfer callback
  if (wr->host_to_device_cb) {
    // while there is an ongoing workrequest, quiescence should not be detected
    // even if all PEs seem idle
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);

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
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);

#ifdef HAPI_CUDA_CALLBACK
    addCallback(wr, AfterKernel);
#else
    recordEvent(wr->stream, wr->kernel_cb, NULL);
#endif
  }

  // transfer data to host
  CsvAccess(gpu_manager).deviceToHostTransfer(wr);

  // add device-to-host transfer callback
  CmiAssert(hapiQdCreate);
  hapiQdCreate(1);
#ifdef HAPI_CUDA_CALLBACK
  // always invoked to free memory
  addCallback(wr, AfterDeviceToHost);
#else
  if (wr->device_to_host_cb) {
    recordEvent(wr->stream, wr->device_to_host_cb, NULL, wr);
  }
  else {
    recordEvent(wr->stream, NULL, NULL, wr);
  }
#endif

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).progress_lock_);
#endif
}

/******************** DEPRECATED ********************/
// Creates a hapiWorkRequest object on the heap and returns it to the user.
hapiWorkRequest* hapiCreateWorkRequest() {
  return (new hapiWorkRequest);
}

hapiWorkRequest::hapiWorkRequest() :
    grid_dim(0), block_dim(0), shared_mem(0), host_to_device_cb(NULL),
    kernel_cb(NULL), device_to_host_cb(NULL), runKernel(NULL), state(0),
    user_data(NULL), free_user_data(false), free_host_to_device_cb(false),
    free_kernel_cb(false), free_device_to_host_cb(false)
  {
#ifdef HAPI_TRACE
    trace_name = "";
#endif
#ifdef HAPI_INSTRUMENT_WRS
    chare_index = -1;
#endif

#if CMK_SMP
    CmiLock(CsvAccess(gpu_manager).stream_lock_);
#endif

    // Create default per-PE streams if none exist
    if (CsvAccess(gpu_manager).getStream(0) == NULL) {
      CsvAccess(gpu_manager).createNStreams(CmiMyNodeSize());
    }

    stream = CsvAccess(gpu_manager).getStream(CmiMyRank() % CsvAccess(gpu_manager).getNStreams());

#if CMK_SMP
    CmiUnlock(CsvAccess(gpu_manager).stream_lock_);
#endif
  }

// Create POSIX shared memory region accessible to all processes on the same host
// Invoked by PE rank 0 of each process (no locking needed for SMP)
void shmCreate() {
  struct stat shm_file_stat;

  // Create the shared memory file
  CsvAccess(gpu_manager).shm_name.assign("cudaipc_shmem-");
  int host_id = CmiPhysicalNodeID(CmiMyPe());
  CsvAccess(gpu_manager).shm_name.append(std::to_string(host_id));
  CsvAccess(gpu_manager).shm_file = shm_open(CsvAccess(gpu_manager).shm_name.c_str(),
      O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
  if (CsvAccess(gpu_manager).shm_file < 0) {
    CmiError("Failure at shm_open");
    goto shm_cleanup;
  }

  // Calculate shared memory region size
  CsvAccess(gpu_manager).shm_chunk_size = sizeof(cudaIpcMemHandle_t) +
      sizeof(cuda_ipc_event_shared) * CsvAccess(gpu_manager).ipc_event_pool_size;
  CsvAccess(gpu_manager).shm_size = CsvAccess(gpu_manager).shm_chunk_size *
    CsvAccess(gpu_manager).device_count_on_physical_node;

  // Set it to the appropriate size
  // Only done by the first process on each physical node
  if (CmiMyNodeRankLocal() == 0) {
    if (ftruncate(CsvAccess(gpu_manager).shm_file, 0) != 0) {
      CmiError("Failure at ftruncate");
      goto shm_cleanup;
    }

    if (ftruncate(CsvAccess(gpu_manager).shm_file, CsvAccess(gpu_manager).shm_size) != 0) {
      CmiError("Failure at ftruncate");
      goto shm_cleanup;
    }
  }

  // Busywait until file is properly sized
  do {
    if (fstat(CsvAccess(gpu_manager).shm_file, &shm_file_stat) != 0) {
      CmiError("Failure at fstat");
      goto shm_cleanup;
    }
  } while (shm_file_stat.st_size != CsvAccess(gpu_manager).shm_size);

  // Load into memory
  CsvAccess(gpu_manager).shm_ptr = mmap(0, CsvAccess(gpu_manager).shm_size,
      PROT_READ | PROT_WRITE, MAP_SHARED, CsvAccess(gpu_manager).shm_file, 0);
  if (CsvAccess(gpu_manager).shm_ptr == (void*)-1) {
    CmiError("Failure at mmap");
    goto shm_cleanup;
  }

  // Store pointer to my process' portion of the shared memory region
  CsvAccess(gpu_manager).shm_my_ptr = (void*)((char*)CsvAccess(gpu_manager).shm_ptr +
      CsvAccess(gpu_manager).shm_chunk_size * CsvAccess(gpu_manager).device_count *
      CmiMyNodeRankLocal());

  // Allocate memory for local storage
  for (int i = 0; i < CsvAccess(gpu_manager).device_count_on_physical_node; i++) {
    CsvAccess(gpu_manager).cuda_ipc_device_infos.emplace_back();
  }

  return;

shm_cleanup:
  shmCleanup();
  CmiAbort("Failure in shared memory region creation");
}

// Clean up shared memory region
// Invoked by PE rank 0 of each process
void shmCleanup() {
  if (CsvAccess(gpu_manager).shm_ptr != NULL) {
    munmap(CsvAccess(gpu_manager).shm_ptr, CsvAccess(gpu_manager).shm_size);
  }

  if (CsvAccess(gpu_manager).shm_file != -1) {
    close(CsvAccess(gpu_manager).shm_file);
  }

  if (!CsvAccess(gpu_manager).shm_name.empty()) {
    shm_unlink(CsvAccess(gpu_manager).shm_name.c_str());
    CsvAccess(gpu_manager).shm_name.clear();
  }
}

// Create CUDA IPC handles and populate shared memory region
// Invoked by all PEs
void ipcHandleCreate() {
  // Only device reps should continue to perform the following operations
  // so that they are done only once per device
  if (!CpvAccess(device_rep)) return;

  // Create CUDA IPC memory handle
  CmiAssert(CsvAccess(gpu_manager).device_managers[CpvAccess(my_device)].comm_buffer);
  cudaIpcMemHandle_t* shm_mem_handle = (cudaIpcMemHandle_t*)((char*)CsvAccess(gpu_manager).shm_my_ptr +
      CsvAccess(gpu_manager).shm_chunk_size * CpvAccess(my_device));
  void* device_ptr = (void*)(CsvAccess(gpu_manager).device_managers[CpvAccess(my_device)].comm_buffer->base_ptr);
  hapiCheck(cudaIpcGetMemHandle(shm_mem_handle, device_ptr));

  // Create CUDA IPC events and corresponding handles
  cuda_ipc_event_shared* shm_event_shared = (cuda_ipc_event_shared*)((char*)shm_mem_handle + sizeof(cudaIpcMemHandle_t));
  int device_index = CsvAccess(gpu_manager).device_count * CmiMyNodeRankLocal() + CpvAccess(my_device);
  cuda_ipc_device_info& my_device_info = CsvAccess(gpu_manager).cuda_ipc_device_infos[device_index];

  my_device_info.event_pool_flags = new int[CsvAccess(gpu_manager).ipc_event_pool_size];
  my_device_info.event_pool_buff_offsets = new size_t[CsvAccess(gpu_manager).ipc_event_pool_size];

  for (int i = 0; i < CsvAccess(gpu_manager).ipc_event_pool_size; i++) {
    cuda_ipc_event_shared* cur_shm_event_shared = shm_event_shared + i;

    my_device_info.event_pool_flags[i] = 0;
    my_device_info.event_pool_buff_offsets[i] = 0;
    my_device_info.src_event_pool.emplace_back();
    my_device_info.dst_event_pool.emplace_back();
    hapiCheck(cudaEventCreateWithFlags(&my_device_info.src_event_pool[i],
          cudaEventDisableTiming | cudaEventInterprocess));
    hapiCheck(cudaEventCreateWithFlags(&my_device_info.dst_event_pool[i],
          cudaEventDisableTiming | cudaEventInterprocess));
    hapiCheck(cudaIpcGetEventHandle(&cur_shm_event_shared->src_event_handle,
          my_device_info.src_event_pool[i]));
    hapiCheck(cudaIpcGetEventHandle(&cur_shm_event_shared->dst_event_handle,
          my_device_info.dst_event_pool[i]));
  }

  // Store device comm buffer ptr in local info (just in case)
  my_device_info.buffer = device_ptr;
}

// Open CUDA IPC handles for accessing other processes' device memory
// Invoked by PE rank 0 of each process
void ipcHandleOpen() {
  for (int i = 0; i < CmiNumNodes() / CmiNumPhysicalNodes(); i++) {
    if (i == CmiMyNodeRankLocal()) continue;

    for (int j = 0; j < CsvAccess(gpu_manager).device_count; j++) {
      int device_index = CsvAccess(gpu_manager).device_count * i + j;
      cuda_ipc_device_info& cur_device_info = CsvAccess(gpu_manager).cuda_ipc_device_infos[device_index];

      // Open memory handle
      cudaIpcMemHandle_t* shm_mem_handle = (cudaIpcMemHandle_t*)((char*)CsvAccess(gpu_manager).shm_ptr
          + CsvAccess(gpu_manager).shm_chunk_size * device_index);
      hapiCheck(cudaIpcOpenMemHandle(&cur_device_info.buffer, *shm_mem_handle,
            cudaIpcMemLazyEnablePeerAccess));

      // Open event handles
      cuda_ipc_event_shared* shm_event_shared =
        (cuda_ipc_event_shared*)((char*)shm_mem_handle + sizeof(cudaIpcMemHandle_t));

      cur_device_info.event_pool_flags = NULL;
      cur_device_info.event_pool_buff_offsets = NULL;

      for (int k = 0; k < CsvAccess(gpu_manager).ipc_event_pool_size; k++) {
        cuda_ipc_event_shared* cur_shm_event_shared = shm_event_shared + k;

        cur_device_info.src_event_pool.emplace_back();
        cur_device_info.dst_event_pool.emplace_back();
        hapiCheck(cudaIpcOpenEventHandle(&cur_device_info.src_event_pool[k],
              cur_shm_event_shared->src_event_handle));
        hapiCheck(cudaIpcOpenEventHandle(&cur_device_info.dst_event_pool[k],
              cur_shm_event_shared->dst_event_handle));
      }
    }
  }
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
  CmiPrintf("[HAPI] start event %d of WR %s, profiling stage %d\n",
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
  Cmiprintf("[HAPI] end event %d of WR %s, profiling stage %d\n",
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
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).inst_lock_);
#endif

  if (CsvAccess(gpu_manager).init_instr_) {
    double tt = CmiWallTimer() - (wr->phase_start_time);
    int index = wr->chare_index;
    char type = wr->comp_type;
    char phase = wr->comp_phase;

    std::vector<hapiRequestTimeInfo> &vec = CsvAccess(gpu_manager).avg_times_[index][type];
    if (vec.size() <= phase) {
      vec.resize(phase+1);
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
        CmiPrintf("[HAPI] invalid event during profileWorkRequestEvent\n");
    }
  }
  else {
    CmiPrintf("[HAPI] instrumentation not initialized!\n");
  }

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).inst_lock_);
#endif
#endif // HAPI_INSTRUMENT_WRS
}

// Create a pool with n_slots slots.
// There are n_buffers[i] buffers for each buffer size corresponding to entry i.
// TODO list the alignment/fragmentation issues with either of two allocation schemes:
// if single, large buffer is allocated for each subpool
// if multiple, smaller buffers are allocated for each subpool
static void createPool(int *n_buffers, int n_slots, std::vector<BufferPool> &pools){
  std::vector<size_t>& mempool_boundaries = CsvAccess(gpu_manager).mempool_boundaries_;

  // initialize pools
  pools.resize(n_slots);
  for (int i = 0; i < n_slots; i++) {
    pools[i].size = mempool_boundaries[i];
    pools[i].head = NULL;
  }

  int device;
  cudaDeviceProp device_prop;
  hapiCheck(cudaGetDevice(&device));
  hapiCheck(cudaGetDeviceProperties(&device_prop, device));

  // divide by # of PEs on physical node and multiply by # of PEs in logical node
  size_t available_memory = device_prop.totalGlobalMem /
                           CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyPe()))
                           * CmiMyNodeSize() * HAPI_MEMPOOL_SCALE;

  // pre-calculate memory per size
  int max_buffers = *std::max_element(n_buffers, n_buffers + n_slots);
  int n_buffers_to_allocate[n_slots];
  memset(n_buffers_to_allocate, 0, sizeof(n_buffers_to_allocate));
  size_t buf_size;
  while (available_memory >= mempool_boundaries[0] + sizeof(BufferPoolHeader)) {
    for (int i = 0; i < max_buffers; i++) {
      for (int j = n_slots - 1; j >= 0; j--) {
        buf_size = mempool_boundaries[j] + sizeof(BufferPoolHeader);
        if (i < n_buffers[j] && buf_size <= available_memory) {
          n_buffers_to_allocate[j]++;
          available_memory -= buf_size;
        }
      }
    }
  }

  // pin the host memory
  for (int i = 0; i < n_slots; i++) {
    buf_size = mempool_boundaries[i] + sizeof(BufferPoolHeader);
    int num_buffers = n_buffers_to_allocate[i];

    BufferPoolHeader* hd;
    BufferPoolHeader* previous = NULL;

    // pin host memory in a contiguous block for a slot
    void* pinned_chunk;
    hapiCheck(cudaMallocHost(&pinned_chunk, buf_size * num_buffers));

    // initialize header structs
    for (int j = num_buffers - 1; j >= 0; j--) {
      hd = reinterpret_cast<BufferPoolHeader*>(reinterpret_cast<unsigned char*>(pinned_chunk)
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

static void releasePool(std::vector<BufferPool> &pools){
  for (int i = 0; i < pools.size(); i++) {
    BufferPoolHeader* hdr = pools[i].head;
    if (hdr != NULL) {
      hapiCheck(cudaFreeHost((void*)hdr));
    }
  }
  pools.clear();
}

static int findPool(size_t size){
  int boundary_array_len = CsvAccess(gpu_manager).mempool_boundaries_.size();
  if (size <= CsvAccess(gpu_manager).mempool_boundaries_[0]) {
    return 0;
  }
  else if (size > CsvAccess(gpu_manager).mempool_boundaries_[boundary_array_len-1]) {
    // create new slot
    CsvAccess(gpu_manager).mempool_boundaries_.push_back(size);

    BufferPool newpool;
    hapiCheck(cudaMallocHost((void**)&newpool.head, size + sizeof(BufferPoolHeader)));
    if (newpool.head == NULL) {
      CmiPrintf("[HAPI (%d)] findPool: failed to allocate newpool %d head, size %zu\n",
             CmiMyPe(), boundary_array_len, size);
      CmiAbort("[HAPI] failed newpool allocation");
    }
    newpool.size = size;
#ifdef HAPI_MEMPOOL_DEBUG
    newpool.num = 1;
#endif
    CsvAccess(gpu_manager).mempool_free_bufs_.push_back(newpool);

    BufferPoolHeader* hd = newpool.head;
    hd->next = NULL;
    hd->slot = boundary_array_len;

    return boundary_array_len;
  }
  for (int i = 0; i < CsvAccess(gpu_manager).mempool_boundaries_.size()-1; i++) {
    if (CsvAccess(gpu_manager).mempool_boundaries_[i] < size &&
        size <= CsvAccess(gpu_manager).mempool_boundaries_[i+1]) {
      return (i + 1);
    }
  }
  return -1;
}

static void* getBufferFromPool(int pool, size_t size){
  BufferPoolHeader* ret;

  if (pool < 0 || pool >= CsvAccess(gpu_manager).mempool_free_bufs_.size()) {
    CmiPrintf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %zu invalid pool\n",
           CmiMyPe(), pool, size);
#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] num: %d\n", CmiMyPe(),
           CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif
    CmiAbort("[HAPI] exiting after invalid pool");
  }
  else if (CsvAccess(gpu_manager).mempool_free_bufs_[pool].head == NULL) {
    BufferPoolHeader* hd;
    hapiCheck(cudaMallocHost((void**)&hd, sizeof(BufferPoolHeader) +
                             CsvAccess(gpu_manager).mempool_free_bufs_[pool].size));
#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %zu expand by 1\n",
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

static void returnBufferToPool(int pool, BufferPoolHeader* hd) {
  hd->next = CsvAccess(gpu_manager).mempool_free_bufs_[pool].head;
  CsvAccess(gpu_manager).mempool_free_bufs_[pool].head = hd;
#ifdef HAPI_MEMPOOL_DEBUG
  CsvAccess(gpu_manager).mempool_free_bufs_[pool].num++;
#endif
}

void* hapiPoolMalloc(size_t size) {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).mempool_lock_);
#endif

  if (!CsvAccess(gpu_manager).mempool_initialized_) {
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
    CsvAccess(gpu_manager).mempool_initialized_ = true;

#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] done creating buffer pool\n", CmiMyPe());
#endif
  }

  int pool = findPool(size);
  void* buf = getBufferFromPool(pool, size);

#ifdef HAPI_MEMPOOL_DEBUG
  CmiPrintf("[HAPI (%d)] hapiPoolMalloc size %zu pool %d left %d\n",
         CmiMyPe(), size, pool,
         CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).mempool_lock_);
#endif

  return buf;
}

void hapiPoolFree(void* ptr) {
  // check if mempool was initialized, just return if not
  if (!CsvAccess(gpu_manager).mempool_initialized_)
    return;

  BufferPoolHeader* hd = ((BufferPoolHeader*)ptr) - 1;
  int pool = hd->slot;

#ifdef HAPI_MEMPOOL_DEBUG
  size_t size = hd->size;
#endif

#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).mempool_lock_);
#endif

  returnBufferToPool(pool, hd);

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).mempool_lock_);
#endif

#ifdef HAPI_MEMPOOL_DEBUG
  CmiPrintf("[HAPI (%d)] hapiPoolFree size %zu pool %d left %d\n",
         CmiMyPe(), size, pool,
         CsvAccess(gpu_manager).mempool_free_bufs_[pool].num);
#endif
}

#ifdef HAPI_INSTRUMENT_WRS
void hapiInitInstrument(int n_chares, int n_types) {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).inst_lock_);
#endif

  if (!CsvAccess(gpu_manager).init_instr_) {
    CsvAccess(gpu_manager).avg_times_.resize(n_chares);
    for (int i = 0; i < n_chares; i++) {
      CsvAccess(gpu_manager).avg_times_[i].resize(n_types);
    }
    CsvAccess(gpu_manager).init_instr_ = true;
  }

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).inst_lock_);
#endif
}

hapiRequestTimeInfo* hapiQueryInstrument(int chare, char type, char phase) {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).inst_lock_);
#endif

  if (phase < CsvAccess(gpu_manager).avg_times_[chare][type].size()) {
    return &CsvAccess(gpu_manager).avg_times_[chare][type][phase];
  }
  else {
    return NULL;
  }

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).inst_lock_);
#endif
}

void hapiClearInstrument() {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).inst_lock_);
#endif

  for (int chare = 0; chare < CsvAccess(gpu_manager).avg_times_.size(); chare++) {
    for (char type = 0; type < CsvAccess(gpu_manager).avg_times_[chare].size(); type++) {
      CsvAccess(gpu_manager).avg_times_[chare][type].clear();
    }
    CsvAccess(gpu_manager).avg_times_[chare].clear();
  }
  CsvAccess(gpu_manager).avg_times_.clear();
  CsvAccess(gpu_manager).init_instr_ = false;

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).inst_lock_);
#endif
}
#endif // HAPI_INSTRUMENT_WRS

// Poll HAPI events stored in the PE's queue. Current strategy is to process
// all successive completed events in the queue starting from the front.
// TODO Maybe we should make one pass of all events in the queue instead,
// since there might be completed events later in the queue.
void hapiPollEvents(void* param, double cur_time) {
#ifndef HAPI_CUDA_CALLBACK
  if (CpvAccess(n_hapi_events) <= 0) return;

  std::queue<hapiEvent>& queue = CpvAccess(hapi_event_queue);
  while (!queue.empty()) {
    hapiEvent hev = queue.front();
    if (cudaEventQuery(hev.event) == cudaSuccess) {
      // invoke Charm++ callback if one was given
      if (hev.cb) {
        CmiAssert(hapiInvokeCallback);
        hapiInvokeCallback(hev.cb, hev.cb_msg);
      }

      // clean up hapiWorkRequest
      if (hev.wr) {
        hapiWorkRequestCleanup(hev.wr);
      }
      cudaEventDestroy(hev.event);
      queue.pop();
      CpvAccess(n_hapi_events)--;

      // inform QD that an event was processed
      CmiAssert(hapiQdProcess);
      hapiQdProcess(1);
    }
    else {
      // stop going through the queue once we encounter a non-successful event
      break;
    }
  }
#endif
}

int hapiCreateStreams() {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).stream_lock_);
#endif

  int ret = CsvAccess(gpu_manager).createStreams();

#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).stream_lock_);
#endif

  return ret;
}

cudaStream_t hapiGetStream() {
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).stream_lock_);
#endif

  cudaStream_t ret = CsvAccess(gpu_manager).getNextStream();

#if CMK_SMP
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
#if CMK_SMP
  CmiLock(CsvAccess(gpu_manager).queue_lock_);
#endif
*/

  // create converse message to be delivered to this PE after CUDA callback
  hapiCallbackMessage* conv_msg = (hapiCallbackMessage*)CmiAlloc(sizeof(hapiCallbackMessage)); // FIXME memory leak?
  conv_msg->rank = CmiMyRank();
  conv_msg->cb = cb;
  conv_msg->cb_msg = cb_msg;
  CmiSetHandler(conv_msg, CsvAccess(gpu_manager).light_cb_idx_);

  // push into CUDA stream
  hapiCheck(cudaLaunchHostFunc(stream, CUDACallback, (void*)conv_msg));

  /*
#if CMK_SMP
  CmiUnlock(CsvAccess(gpu_manager).queue_lock_);
#endif
*/
#endif

  // while there is an ongoing workrequest, quiescence should not be detected
  // even if all PEs seem idle
  CmiAssert(hapiQdCreate);
  hapiQdCreate(1);
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

cudaError_t hapiMallocHostPool(void** ptr, size_t size) {
  void* tmp_ptr = hapiPoolMalloc(size);
  if (tmp_ptr) {
    *ptr = tmp_ptr;
    return cudaSuccess;
  }
  else return cudaErrorMemoryAllocation;
}

cudaError_t hapiFreeHost(void* ptr) {
  return cudaFreeHost(ptr);
}

cudaError_t hapiFreeHostPool(void *ptr) {
  hapiPoolFree(ptr);
  return cudaSuccess;
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
